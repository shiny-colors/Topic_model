#####相関トピックモデル#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
detach("package:gtools", unload=TRUE)
library(bayesm)
library(ExtDist)
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
  eigen(x.modified)
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
#文書データの設定
k <- 8   #トピック数
d <- 2000   #文書数
v <- 300   #語彙数
w <- rpois(d, rgamma(d, 160, 1.0))   #1文書あたりの単語数

#IDの設定
word_id <- rep(1:d, w)

##パラメータの設定
alpha1 <- rep(0.25, v)   #単語のディレクリ事前分布のパラメータ

#多変量正規分布のパラメータを設定
alpha0 <- rep(0, k-1)   #文書ののパラメータ
tau0 <- corrM(k-1, -0.6, 0.9, 0.01, 0.2)
sigma0 <- covmatrix(k-1, tau0, 2.5, 3.0)$covariance

#多変量正規分布からの乱数を多項ロジット変換して文書トピックを設定
mv <- cbind(mvrnorm(d, alpha0, sigma0), 0)
theta0 <- exp(mv) / rowSums(exp(mv))

#ディレクリ乱数の発生
phi0 <- extraDistr::rdirichlet(k, alpha1)   #単語のトピック分をディレクリ分布から発生

##多項分布からトピックおよび単語データを発生
WX <- matrix(0, nrow=d, ncol=v)
Z1 <- list()

#文書ごとにトピックと単語を逐次生成
for(i in 1:d){
  print(i)

  #文書のトピック分布を発生
  z1 <- t(rmultinom(w[i], 1, theta0[i, ]))   #文書のトピック分布を発生
  
  #文書のトピック分布から単語を発生
  zn <- z1 %*% c(1:k)   #0,1を数値に置き換える
  zdn <- cbind(zn, z1)   #apply関数で使えるように行列にしておく
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi0[x[1], ])))   #文書のトピックから単語を生成
  wdn <- colSums(wn)   #単語ごとに合計して1行にまとめる
  WX[i, ] <- wdn  
  
  #発生させたトピックを格納
  Z1[[i]] <- zdn[, 1]
}
storage.mode(WX) <- "integer"   #データ行列を整数型行列に変更


####トピックモデルのためのデータと関数の準備####
##それぞれの文書中の単語の出現および補助情報の出現をベクトルに並べる
##データ推定用IDを作成
ID_list <- list()
wd_list <- list()

#求人ごとに求人IDおよび単語IDを作成
for(i in 1:nrow(WX)){
  print(i)
  
  #単語のIDベクトルを作成
  ID_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * (1:v)
  num2 <- subset(num1, num1 > 0)
  W1 <- WX[i, (WX[i, ] > 0)]
  number <- rep(num2, W1)
  wd_list[[i]] <- number
}

#リストをベクトルに変換
ID_d <- unlist(ID_list)
wd <- unlist(wd_list)

##インデックスを作成
doc_list <- list()
word_list <- list()
for(i in 1:length(unique(ID_d))) {doc_list[[i]] <- subset(1:length(ID_d), ID_d==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- subset(1:length(wd), wd==i)}
gc(); gc()

x %*% etad[4, ]
t(x %*% t(etad))[4, ]

####マルコフ連鎖モンテカルロ法で対応トピックモデルを推定####
##多項ロジットモデルの対数尤度関数
loglike <- function(beta, y, X, N, select){
  
  #ロジットと確率の計算
  logit <- t(X %*% t(beta))
  Pr <- exp(logit) / matrix(rowSums(exp(logit)), nrow=N, ncol=select)
  
  #対数尤度を定義
  LLi <- rowSums(y * log(Pr))
  LL <- sum(LLi)
  val <- list(LLi=LLi, LL=LL)
  return(val)
}

##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #負担係数の格納用
  for(kk in 1:k){
    #負担係数を計算
    Bi <- rep(theta[, kk], w) * phi[kk, c(wd)]   #尤度
    Bur[, kk] <- Bi   
  }
  
  Br <- Bur / rowSums(Bur)   #負担率の計算
  r <- colSums(Br) / sum(Br)   #混合率の計算
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##アルゴリズムの設定
R <- 10000   #サンプリング回数
keep <- 2   #2回に1回の割合でサンプリング結果を格納
iter <- 0

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- rep(0, k-1)
nu <- k+1
V <- nu * diag(k-1)
alpha02 <- rep(0.5, v)
beta0m <- matrix(1, nrow=v, ncol=k)

##パラメータの初期値
#多変量正規分布からトピック分布を発生
oldalpha <- rep(0, k-1)
oldcov <- diag(k-1)
cov_inv <- solve(oldcov)
oldeta <- cbind(mvrnorm(d, oldalpha, oldcov), 1)
theta <- exp(mv) / rowSums(exp(mv))

#ディクレリ分布から単語分布を発生
phi.ini <- runif(v, 0.5, 1)
phi <- extraDistr::rdirichlet(k, phi.ini)   #単語トピックのパラメータの初期値


##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
SIGMA <- array(0, dim=c(k-1, k-1, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
Z_SEG <- matrix(0, nrow=length(wd), ncol=k)
storage.mode(Z_SEG) <- "integer"
gc(); gc()


##MCMC推定用配列
wsum0 <- matrix(0, nrow=d, ncol=k)
vf0 <- matrix(0, nrow=v, ncol=k)
x <- diag(k)[, -k]
lognew <- rep(0, d)
logold <- rep(0, d)
logpnew <- rep(0, d)
logpold <- rep(0, d)


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){

  ##単語ごとにトピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_rate <- burden_fr(theta, phi, wd, w, k)$Br
  
  #多項分布から単語トピックをサンプリング
  Zi1 <- rmnom(nrow(word_rate), 1, word_rate)
  word_z <- as.numeric(Zi1 %*% 1:k)
  
  ##メトロポリスヘイスティング法で単語トピックのパラメータを更新
  #多項ロジットモデルの応答変数を生成
  for(i in 1:d){wsum0[i, ] <- colSums(Zi1[doc_list[[i]], ])}
  
  #相関のある多項ロジットモデルのパラメータのサンプリング
  #新しいパラメータをサンプリング
  etad <- oldeta[, -k]
  etan <- etad + matrix(rnorm(d*(k-1), 0, 0.1), nrow=d, ncol=k-1)
  
  #事前分布の誤差を計算
  er_new <- etan - matrix(0, nrow=d, ncol=k-1, byrow=T)
  er_old <- etad - matrix(0, nrow=d, ncol=k-1, byrow=T)

  #対数尤度と対数事前分布を計算
  lognew <- loglike(etan, wsum0, x, d, k)$LLi
  logold <- loglike(etad, wsum0, x, d, k)$LLi
  logpnew <- apply(er_new, 1, function(x) -0.5 * (x %*% cov_inv %*% x))
  logpold <- apply(er_old, 1, function(x) -0.5 * (x %*% cov_inv %*% x))
  
  #メトロポリスヘイスティング法でパラメータの採択を決定
  rand <- runif(d)   #一様分布から乱数を発生
  LLind_diff <- exp(lognew + logpnew - logold - logpold)   #採択率を計算
  alpha <- (LLind_diff > 1)*1 + (LLind_diff <= 1)*LLind_diff
  
  #alphaの値に基づき新しいbetaを採択するかどうかを決定
  flag <- matrix(((alpha >= rand)*1 + (alpha < rand)*0), nrow=d, ncol=k-1)
  oldeta <- flag*etan + (1-flag)*etad   #alphaがrandを上回っていたら採択
  
  #多項ロジットモデルからthetaを更新
  oldeta0 <- cbind(oldeta, 0)
  theta <- exp(oldeta0) / rowSums(exp(oldeta0))
  
  ##逆ウィシャート分布から分散共分散行列をサンプリング
  V_par <- d + nu
  R_par <- solve(V) + t(oldeta) %*% oldeta
  oldcov <- rwishart(V_par, solve(R_par))$IW
  cov_inv <- solve(oldcov)

  ##ディクレリ分布からphiをサンプリング
  for(i in 1:v){vf0[i, ] <- colSums(Zi1[word_list[[i]], ])}
  vf <- vf0 + beta0m
  phi <- extraDistr::rdirichlet(k, t(vf))
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep1 <- rp/keep
    THETA[, , mkeep1] <- theta
    PHI[, , mkeep1] <- phi
    SIGMA[, , mkeep1] <- cov2cor(oldcov)
    
    #トピック割当はサンプリング期間の半分を超えたら格納する
    if(rp >= R/2){
      Z_SEG <- Z_SEG + Zi1
    }
  
    #サンプリング結果を確認
    print(rp)
    print(sum(lognew))
    print(round(cbind(cov2cor(oldcov), cov2cor(sigma0)), 3))
    print(round(rbind(theta[1:5, ], theta0[1:5, ]), 3))
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




