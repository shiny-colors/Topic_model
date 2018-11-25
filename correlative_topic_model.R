#####相関トピックモデル#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(842573)

####多変量正規分布の乱数を発生させる関数を定義####
#任意の相関行列を作る関数を定義
corrM <- function(col, lower, upper, eigen_lower, eigen_upper){
  diag(1, col, col)
  
  rho <- matrix(runif(col^2, lower, upper), col, col)
  rho[upper.tri(rho)] <- 0
  Sigma <- rho + t(rho)
  diag(Sigma) <- 1
  (X.Sigma <- eigen(Sigma))
  (Lambda <- diag(X.Sigma$values))
  P <- X.Sigma$vector
  
  #新しい相関行列の定義と対角成分を1にする
  (Lambda.modified <- ifelse(Lambda < 0, runif(1, eigen_lower, eigen_upper), Lambda))
  x.modified <- P %*% Lambda.modified %*% t(P)
  normalization.factor <- matrix(diag(x.modified),nrow = nrow(x.modified),ncol=1)^0.5
  Sigma <- x.modified <- x.modified / (normalization.factor %*% t(normalization.factor))
  diag(Sigma) <- 1
  round(Sigma, digits=3)
  return(Sigma)
}


##相関行列から分散共分散行列を作成する関数を定義
covmatrix <- function(col, corM, lower, upper){
  m <- abs(runif(col, lower, upper))
  c <- matrix(0, col, col)
  for(i in 1:col){
    for(j in 1:col){
      c[i, j] <- sqrt(m[i]) * sqrt(m[j])
    }
  }
  diag(c) <- m
  cc <- c * corM
  
  #固有値分解で強制的に正定値行列に修正する
  UDU <- eigen(cc)
  val <- UDU$values
  vec <- UDU$vectors
  D <- ifelse(val < 0, val + abs(val) + 0.00001, val)
  covM <- vec %*% diag(D) %*% t(vec)
  data <- list(covM, cc,  m)
  names(data) <- c("covariance", "cc", "mu")
  return(data)
}

####データの発生####
#set.seed(423943)
##データの設定
k <- 15   #トピック数
d <- 3000   #文書数
v <- 1200   #語彙数
w <- rpois(d, rgamma(d, 35, 0.25))   #1文書あたりの単語数
f <- sum(w)   #総単語数

##IDとインデックスの設定
#IDの設定
d_id <- rep(1:d, w)
t_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))

#インデックスの設定
d_index <- list()
for(i in 1:d){
  d_index[[i]] <- which(d_id==i)
}

##単語がすべて生成されるまで繰り返す
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  ##パラメータの生成
  #事前分布の設定
  alpha <- rep(0.05, v)   #単語の事前分布のパラメータ
  
  #多変量正規分布のパラメータを設定
  mu <- rep(0, k-1)   #文書のパラメータ
  tau <- corrM(k-1, -0.6, 0.9, 0.01, 0.2)
  Cov <- Covt <- covmatrix(k-1, tau, 4.0, 4.0)$covariance
  
  #多変量正規分布からの乱数を多項ロジット変換して文書トピックを設定
  beta <- betat <- cbind(mvrnorm(d, mu, Cov), 0)
  theta <- thetat <- exp(beta) / rowSums(exp(beta))
  
  #単語分布をディレクリ分布から発生
  phi <- extraDistr::rdirichlet(k, alpha)   
  
  #出現確率が低いphiの要素を入れ替える
  index <- which(colMaxs(phi) < (k*5)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(2.5, k))) %*% 1:k), index[j]] <- (k*5)/f
  }
  phit <- phi
  
  ##トピックおよび単語データを発生
  WX <- matrix(0, nrow=d, ncol=v)
  ZX <- matrix(0, nrow=d, ncol=k)
  word_list <- list()
  
  #多項分布からトピックを生成
  Z <- rmnom(f, 1, theta[d_id, ])
  z_vec <- as.numeric(Z %*% 1:k)
  
  #トピックから単語を生成
  for(i in 1:d){
    index <- d_index[[i]]
    word <- rmnom(w[i], 1, phi[z_vec[index], ])
    
    #パラメータを格納
    ZX[i, ] <- colSums(Z[index, ])
    WX[i, ] <- colSums(word)
    word_list[[i]] <- as.numeric(word %*% 1:v)
  }
  #break条件
  if(min(colSums(WX)) > 0){
    break
  }
}

#リストを変換
wd <- unlist(word_list)
sparse_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
sparse_data_T <- t(sparse_data)


####マルコフ連鎖モンテカルロ法でCorrelative Topic modelを推定####
#対数事後分布を計算する関数
loglike <- function(zsum, beta, mu, inv_Cov, k){
  
  #ロジットモデルの対数尤度
  par <- cbind(beta, 0)
  logit_exp <- exp(par)   #ロジットの期待値の指数
  prob <- logit_exp / as.numeric(logit_exp %*% rep(1, k))   #選択確率
  Li <- as.numeric((zsum * log(prob)) %*% rep(1, k))
  
  #多変量正規分布の対数事前分布
  er <- beta - matrix(mu, nrow=d, ncol=k-1, byrow=T)
  log_prior <- -1/2 * as.numeric((er %*% inv_Cov * er) %*% rep(1, k-1))
  
  #ユーザーごとの対数尤度
  LLi <- Li + log_prior
  return(LLi)
}

#対数事後分布の微分関数
dloglike <- function(zsum_vec, Data, beta, mu, inv_Cov, z_dt, k){
  
  #応答確率の設定
  par <- cbind(beta, 0)
  logit_exp <- exp(par)   #ロジットの期待値の指数
  prob <- logit_exp / as.numeric(logit_exp %*% rep(1, k))   #選択確率
  prob_vec <- as.numeric(t(prob))
  
  #微分関数の設定
  er <- beta - matrix(mu, nrow=d, ncol=k-1, byrow=T)
  dlogit <- (zsum_vec - prob_vec) * Data   #ロジットモデルの対数尤度の微分関数
  dmvn <- -t(inv_Cov %*% t(er))

  #対数事後分布の微分関数の和
  LLd <- -(z_dt %*% dlogit + dmvn)
  return(LLd)
}

#リープフロッグ法を解く関数
leapfrog <- function(r, z, D, e, L) {
  leapfrog.step <- function(r, z, e){
    r2 <- r  - e * D(wsum_vec, Data, z, mu, inv_Cov, z_dt, k) / 2
    z2 <- z + e * r2
    r2 <- r2 - e * D(wsum_vec, Data, z2, mu, inv_Cov, z_dt, k) / 2
    list(r=r2, z=z2) # 1回の移動後の運動量と座標
  }
  leapfrog.result <- list(r=r, z=z)
  for(i in 1:L) {
    leapfrog.result <- leapfrog.step(leapfrog.result$r, leapfrog.result$z, e)
  }
  leapfrog.result
}

##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / rowSums(Bur)   #負担率
  r <- colSums(Br) / sum(Br)   #混合率
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##アルゴリズムの設定
LL1 <- -100000000   #対数尤度の初期値
R <- 5000
keep <- 2  
iter <- 0
burnin <- 500/keep
disp <- 10
e <- 0.0025
L <- 5

##データとインデックスの設定
#データの設定
Data <- matrix(diag(k), nrow=d*k, ncol=k, byrow=T)[, -k]

#インデックスの設定
w_list <- list()
for(j in 1:v){
  w_list[[j]] <- which(wd==j)
}
w_dt <- sparseMatrix(wd, 1:f, x=rep(1, f), dims=c(v, f))
d_dt <- sparseMatrix(d_id, 1:f, x=rep(1, f), dims=c(d, f))
z_dt <- sparseMatrix(rep(1:d, rep(k, d)), 1:(k*d), x=rep(1, k*d), dims=c(d, k*d))

##事前分布の設定
#多変量正規分布の事前分布
mu <- rep(0, k-1)
nu <- k - 1
V <- solve(nu * diag(k-1))

#単語分布の事前分布
alpha <- 0.1


##パラメータの真値
#多変量正規分布の真値
beta <- betat 
Cov <- Covt; inv_Cov <- solve(Cov)

#トピックモデルの真値
theta <- thetat
phi <- phit

##パラメータの初期値
#多変量正規分布の初期値
beta <- mvrnorm(d, rep(0, k-1), diag(k-1))
Cov <- diag(k-1); inv_Cov <- solve(Cov)

#トピックモデルの初期値
theta <- extraDistr::rdirichlet(d, rep(2.5, k))
phi <- extraDistr::rdirichlet(k, rep(2.5, v))

##パラメータの格納用配列
BETA <- array(0, dim=c(d, k-1, R/keep))
THETA <- array(0, dim=c(d, k, R/keep))
COV <- array(0, dim=c(k-1, k-1, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG) <- "integer"
gc(); gc()


##対数尤度の基準値
#ユニグラムモデルの対数尤度
LLst <- sum(sparse_data %*% log(colSums(WX) / sum(WX)))

#ベストなパラメータの対数尤度
LLbest <- sum(log(rowSums(thetat[d_id, ] * t(phit)[wd, ])))


####ハミルトニアンモンテカルロ法でパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとにトピックをサンプリング
  #トピックの割当確率
  Lho <- theta[d_id, ] * t(phi)[wd, ]
  topic_rate <- Lho / as.numeric(Lho %*% rep(1, k))
  
  #多項分布から単語トピックをサンプリング
  Zi <- rmnom(f, 1, topic_rate)
  z_vec <- as.numeric(Zi %*% 1:k)
  
  
  ##ロジットモデルのパラメータサンプリング
  #ロジットモデルの応答変数を設定
  wsum <- d_dt %*% Zi
  wsum_vec <- as.numeric(t(wsum))
  
  #新しいパラメータをサンプリング
  rold <- mvrnorm(d, rep(0, k-1), diag(k-1))   #標準多変量正規分布からパラメータを生成
  betad <- beta

  #リープフロッグ法による1ステップ移動
  res <- leapfrog(rold, betad, dloglike, e, L)
  rnew <- res$r
  betan <- res$z
  
  #移動前と移動後のハミルトニアン
  Hnew <- -loglike(wsum, betan, mu, inv_Cov, k) + as.numeric(rnew^2 %*% rep(1, k-1))/2
  Hold <- -loglike(wsum, betad, mu, inv_Cov, k) + as.numeric(rold^2 %*% rep(1, k-1))/2
  
  #パラメータの採択を決定
  rand <- runif(d) #一様分布から乱数を発生
  gamma <- rowMins(cbind(1, exp(Hold - Hnew)))   #採択率を決定
  gamma_mu <- mean(gamma)
  
  #alphaの値に基づき新しいbetaを採択するかどうかを決定
  flag <- as.numeric(gamma > rand)
  beta <- as.matrix(flag*betan + (1-flag)*betad)

  
  #パラメータを確率に変換
  par <- exp(cbind(beta, 0))
  theta <- par / rowSums(par) 
  
  ##逆ウィシャート分布から分散共分散行列をサンプリング
  er <- beta - matrix(mu, nrow=d, ncol=k-1, byrow=T)
  V_par <- d + nu
  R_par <- t(er) %*% er + V
  Cov <- rwishart(V_par, solve(R_par))$IW
  inv_Cov <- solve(Cov)
  
  
  ##ディクレリ分布からphiをサンプリング
  vsum <- t(w_dt %*% Zi) + alpha   #ディリクレ分布のパラメータ
  phi <- extraDistr::rdirichlet(k, vsum)   
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    BETA[, , mkeep] <- beta
    COV[, , mkeep] <- Cov
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      SEG <- SEG + Zi
    }
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL <- sum(log(rowSums(Lho)))
    
    #サンプリング結果を確認
    print(rp)
    print(gamma_mu)
    print(c(LL, LLbest, LLst))
    print(sum(loglike(wsum, beta, mu, inv_Cov, k)))
    print(round(cov2cor(Cov[1:7, 1:7]), 2))
    print(round(rbind(theta[1:5, ], thetat[1:5, ]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 2000   #バーンイン期間

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA[1, , ]), type="l", ylab="パラメータ", main="文書1のトピック分布のサンプリング結果")
matplot(t(THETA[2, , ]), type="l", ylab="パラメータ", main="文書2のトピック分布のサンプリング結果")
matplot(t(THETA[3, , ]), type="l", ylab="パラメータ", main="文書3のトピック分布のサンプリング結果")
matplot(t(THETA[4, , ]), type="l", ylab="パラメータ", main="文書4のトピック分布のサンプリング結果")

#分散共分散行列のサンプリング結果
matplot(t(SIGMA[1,  , ]), type="l", ylab="パラメータ")
matplot(t(SIGMA[2,  , ]), type="l", ylab="パラメータ")
matplot(t(SIGMA[3,  , ]), type="l", ylab="パラメータ")
matplot(t(SIGMA[4,  , ]), type="l", ylab="パラメータ")
matplot(t(SIGMA[5,  , ]), type="l", ylab="パラメータ")
matplot(t(SIGMA[6,  , ]), type="l", ylab="パラメータ")
matplot(t(SIGMA[7,  , ]), type="l", ylab="パラメータ")

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, 1:10, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[2, 11:20, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[3, 21:30, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[4, 31:40, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(PHI[5, 41:50, ]), type="l", ylab="パラメータ", main="トピック5の単語の出現率のサンプリング結果")
matplot(t(PHI[6, 51:60, ]), type="l", ylab="パラメータ", main="トピック6の単語の出現率のサンプリング結果")
matplot(t(PHI[7, 61:70, ]), type="l", ylab="パラメータ", main="トピック7の単語の出現率のサンプリング結果")
matplot(t(PHI[8, 71:80, ]), type="l", ylab="パラメータ", main="トピック8の単語の出現率のサンプリング結果")


##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, theta0), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(rbind(word_mu, phi0)[, 1:50], 3)

#分散共分散行列の事後推定量
sigma_mu <- apply(SIGMA[, , burnin:(R/keep)], c(1, 2), mean)   #分散共分散行列の出現率の事後平均
round(rbind(sigma_mu, sigma0), 3)




