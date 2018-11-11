#####ノイズあり対応トピックモデル#####
options(warn=0)
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(data.table)
library(bayesm)
library(HMM)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(2506787)

####データの生成####
#set.seed(423943)
#文書データの設定
k <- 15   #トピック数
vec_k <- rep(1, k)
d <- 5000   #文書数
v <- 1200   #語彙数
w <- rpois(d, rgamma(d, 40, 0.2))   #1文書あたりの単語数
f <- sum(w)   #総単語数
a1 <- 200   #トピックに関係のあるタグ数
a2 <- 100   #トピックに関係のないタグ数
a <- a1 + a2   #補助変数数
x <- rtpois(d, 50, 2, Inf)
f1 <- sum(w)
f2 <- sum(x)

#IDの設定
w_id <- rep(1:d, w)
a_id <- rep(1:d, x)

#パラメータの設定
#トピック分布を設定
alpha0 <- rep(0.15, k)   #文書のディレクリ事前分布のパラメータ
beta0 <- rbeta(sum(x), 0.55, 0.175)

#単語分布を設定
index <- apply(rmnom(v, f, rep(1, k)), 1, which.max)
alpha1 <- rep(0.04, v)
alpha2 <- c(rep(0.1, a1), rep(0.0001, a2))   #トピックに関係のあるタグのディクレリ事前分布のパラメータ
alpha3 <- c(rep(0.0001, a1), rep(1.0, a2))   #トピックに関係のないタグのディクレリ事前分布のパラメータ


##モデルに基づき単語を生成
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  #ディレクリ分布からパラメータを生成
  thetat <- theta <- extraDistr::rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から生成
  phit <- phi <- extraDistr::rdirichlet(k, alpha1)   #単語のトピック分布をディレクリ乱数から生成
  lambda <- matrix(0, nrow=d, ncol=k)   #文書に含むトピックだけを補助情報のトピックにするための確率を格納する行列
  omegat <- omega <- extraDistr::rdirichlet(k, alpha2)   #補助情報のトピック分布をディクレリ乱数から生成
  gammat <- gamma <- extraDistr::rdirichlet(1, alpha3)   #トピックに関係のないタグ
  omega0 <- rbind(omega, gamma)   #単語分布の結合
  
  #単語出現確率が低いトピックを入れ替える
  index <- which(colMaxs(phi) < (k*10)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, alpha0)) %*% 1:k), index[j]] <- (k*10)/f
  }
  phit <- phi
  
  ##多項分布からトピックおよび単語データを生成
  WX <- matrix(0, nrow=d, ncol=v)
  AX <- matrix(0, nrow=d, ncol=a)
  word_list <- list()
  aux_list <- list()
  Z0 <- rep(0, sum(x)) 
  Z1_list <- list()
  Z2_list <- list()
  
  #文書ごとにトピックと単語を逐次生成
  for(i in 1:d){

    #文書のトピック分布を生成
    z1 <- rmnom(w[i], 1, theta[i, ])   #文書のトピック分布を生成
    z1_vec <- as.numeric(z1 %*% 1:k)
    
    #文書のトピック分布から単語を生成
    word <- rmnom(w[i], 1, phi[z1_vec, ])   #文書のトピックから単語を生成
    word_vec <- colSums(word)   #単語ごとに合計して1行にまとめる
    WX[i, ] <- word_vec
    
    #文書のトピック分布から補助変数を生成
    #文書で生成させたトピックのみを補助情報のトピック分布とする
    lambda[i, ] <- colSums(z1) / w[i]   #補助情報のトピック分布
    
    #ベルヌーイ分布からトピックに関係があるかどうかを生成
    index <- which(a_id==i)
    Z0[index] <- rbinom(length(index), 1, beta0[index])
  
    #補助情報のトピックを生成
    z2_aux <- rmnom(x[i], 1, lambda[i, ])
    z2 <- cbind(z2_aux * Z0[index], 1-Z0[index])
    z2_vec <- as.numeric(z2 %*% 1:(k+1))
    
    #生成させたトピックの単語分布に従い単語を生成
    aux <- rmnom(x[i], 1, omega0[z2_vec, ])
    aux_vec <- colSums(aux)
    AX[i, ] <- aux_vec
    
    #文書トピックおよび補助情報トピックを格納
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
    word_list[[i]] <- as.numeric(word %*% 1:v)
    aux_list[[i]] <- as.numeric(aux %*% 1:a)
  }
  if(min(colSums(AX)) > 0 & min(colSums(WX)) > 0){
    break
  }
}

#データ行列を整数型行列に変更
Z1 <- do.call(rbind, Z1_list)
Z2 <- do.call(rbind, Z2_list)
wd <- unlist(word_list)
ad <- unlist(aux_list)
storage.mode(WX) <- "integer"
storage.mode(AX) <- "integer"
r0 <- c(mean(Z0), 1-mean(Z0))

#単語ベクトルを行列化
word_data <- sparseMatrix(1:f1, wd, x=rep(1, f1), dims=c(f1, v))
aux_data <- sparseMatrix(1:f2, ad, x=rep(1, f2), dims=c(f2, a))


####マルコフ連鎖モンテカルロ法で対応トピックモデルを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / as.numeric(Bur %*% vec_k)   #負担率
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}


##アルゴリズムの設定
R <- 3000   #サンプリング回数
keep <- 2   #2回に1回の割合でサンプリング結果を格納
disp <- 10
iter <- 0
burnin <- 1000/keep

##インデックスを設定
d_dt <- sparseMatrix(w_id, 1:f1, x=rep(1, f1), dims=c(d, f1))
a_dt <- sparseMatrix(a_id, 1:f2, x=rep(1, f2), dims=c(d, f2))
word_dt <- t(word_data)
aux_dt <- t(aux_data)

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 0.25
alpha02 <- 0.25
beta01 <- 0.01
beta02 <- 0.01
s0 <- 0.01
v0 <- 0.01

##パラメータの真値
theta <- thetat
phi <- phit
omega <- omegat
gamma <- gammat
r <- 0.5

##パラメータの初期値
theta <- extraDistr::rdirichlet(d, rep(1, k))   #文書トピックのパラメータの初期値
phi <- extraDistr::rdirichlet(k, rep(2.0, v))   #単語トピックのパラメータの初期値
omega <- extraDistr::rdirichlet(k, rep(10, a))   #タグのトピックのパラメータの初期値
gamma <- as.numeric(extraDistr::rdirichlet(1, rep(10, a)))   #内容と関係のタグのパラメータの初期値
r <- 0.5   #内容に関係があるかどうかの混合率

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
OMEGA <- array(0, dim=c(k, a, R/keep))
GAMMA <- matrix(0, nrow=R/keep, ncol=a)
LAMBDA <- rep(0, R/keep)
Z_SEG <- rep(0, f2)
W_SEG <- matrix(0, nrow=f1, ncol=k)
A_SEG <- matrix(0, nrow=f2, ncol=k+1)
storage.mode(W_SEG) <- "integer"
storage.mode(A_SEG) <- "integer"
storage.mode(Z_SEG) <- "integer"
gc(); gc()

##対数尤度の基準値
LLst1 <- sum(log((colSums(WX) / f1)[wd]))
LLst2 <- sum(log((colSums(AX) / f2)[ad]))
LLst <- LLst1 + LLst2


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語トピックをサンプリング
  #単語ごとにトピックの出現確率を計算
  word_par <- burden_fr(theta, phi, wd, w_id, k, vec_k)
  word_rate <- word_par$Br
  
  #多項分布から単語トピックをサンプリング
  Zi1 <- rmnom(f1, 1, word_rate)   
  z1_vec <- as.numeric(Zi1 %*% 1:k)
  
  
  ##単語トピックのパラメータを更新
  #ディクレリ分布からthetaをサンプリング
  wsum0 <- as.matrix(d_dt %*% Zi1)
  wsum <- wsum0 + alpha01   #ディクレリ分布のパラメータ
  theta <- extraDistr::rdirichlet(d, wsum)   #パラメータをサンプリング
  
  #ディクレリ分布からphiをサンプリング
  vf <- as.matrix(t(word_dt %*% Zi1)) + beta01   #ディクレリ分布のパラメータ
  phi <- extraDistr::rdirichlet(k, vf)   #パラメータをサンプリング
  
  
  ##タグが文書のトピックと関連があるかどうかを決定
  #生成させた単語トピックから補助情報のトピック分布を設定
  theta_aux <- wsum0 / w
  
  #ベルヌーイ分布の確率を計算
  aux_par <- theta_aux[a_id, ] * t(omega)[ad, ]   #補助情報トピックの尤度
  aux_sums <- as.numeric(aux_par %*% vec_k)
  tau01 <- r * aux_sums   #補助情報トピックの割当尤度
  tau02 <- (1-r) * gamma[ad]   #ノイズ補助変数の割当尤度
  tau <- tau01 / (tau01 + tau02)  
  
  #ベルヌーイ分布よりノイズの潜在変数を生成
  z <- rbinom(f2, 1, tau)
  
  #ベータ分布から混合率をサンプリング
  n <- sum(z)
  s1 <- n + s0
  v1 <- f2 - n + v0
  r <- rbeta(1, s1, v1)   #混合率をサンプリング
  
  ##補助情報トピックをサンプリング
  #z=1の場合、タグごとにトピックの出現率を計算
  index_z <- which(z==1)   #z=1のみ抽出
  aux_rate <- aux_par[index_z, ] / aux_sums[index_z]
  
  #多項分布から補助情報トピックをサンプリング
  Zi2 <- matrix(0, nrow=f2, ncol=k+1)
  Zi2[index_z, 1:k] <- rmnom(n, 1, aux_rate)   #多項分布からトピックをサンプリング
  Zi2[-index_z, k+1] <- 1
  z2_vec <- as.numeric(Zi2 %*% 1:(k+1))
  
  
  ##タグトピックのパラメータを更新
  af <- as.matrix(t(aux_dt %*% Zi2[, 1:k])) + beta02   #ディクレリ分布のパラメータ
  omega <- extraDistr::rdirichlet(k, af)   #パラメータをサンプリング
  
  ##内容に関係のないタグのパラメータの更新
  nf <- as.numeric(aux_dt %*% Zi2[, k+1]) + beta02   #ディクレリ分布のパラメータ
  gamma <- as.numeric(extraDistr::rdirichlet(1, nf))   #パラメータをサンプリング
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    OMEGA[, , mkeep] <- omega
    GAMMA[mkeep, ] <- gamma
    LAMBDA[mkeep] <- r
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      W_SEG <- W_SEG + Zi1
      A_SEG <- A_SEG + Zi2
      Z_SEG <- Z_SEG + z
    }
  }
  
  if(rp%%disp==0){
    #対数尤度の計算
    LL1 <- sum(log(word_par$Bur %*% vec_k))
    LL2 <- sum(log(aux_par[index_z, ] %*% vec_k)) + sum(log(gamma[ad[-index_z]]))
    LL <- LL1 + LL2
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL, LLst, LL1, LL2, LLst1, LLst2))
    print(round(c(r, r0[1]), 3))
    print(round(cbind(omega[, (a1-4):(a1+5)], omegat[, (a1-4):(a1+5)]), 3))
    print(round(rbind(gamma[(a1-9):(a1+10)], gammat[(a1-9):(a1+10)]), 3))
  }
}


####サンプリング結果の可視化と要約####
burnin <- 500   #バーンイン期間

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA[1, , ]), type="l", ylab="パラメータ", main="文書1のトピック分布のサンプリング結果")
matplot(t(THETA[2, , ]), type="l", ylab="パラメータ", main="文書2のトピック分布のサンプリング結果")
matplot(t(THETA[3, , ]), type="l", ylab="パラメータ", main="文書3のトピック分布のサンプリング結果")
matplot(t(THETA[4, , ]), type="l", ylab="パラメータ", main="文書4のトピック分布のサンプリング結果")

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, , ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[2, , ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[3, , ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[4, , ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(PHI[5, , ]), type="l", ylab="パラメータ", main="トピック5の単語の出現率のサンプリング結果")
matplot(t(PHI[6, , ]), type="l", ylab="パラメータ", main="トピック6の単語の出現率のサンプリング結果")
matplot(t(PHI[7, , ]), type="l", ylab="パラメータ", main="トピック7の単語の出現率のサンプリング結果")
matplot(t(PHI[8, , ]), type="l", ylab="パラメータ", main="トピック8の単語の出現率のサンプリング結果")

#タグの出現確率のサンプリング結果
matplot(t(OMEGA[1, , ]), type="l", ylab="パラメータ", main="トピック1のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[2, , ]), type="l", ylab="パラメータ", main="トピック2のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[3, , ]), type="l", ylab="パラメータ", main="トピック3のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[4, , ]), type="l", ylab="パラメータ", main="トピック4のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[5, , ]), type="l", ylab="パラメータ", main="トピック5のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[6, , ]), type="l", ylab="パラメータ", main="トピック6のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[7, , ]), type="l", ylab="パラメータ", main="トピック7のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[8, , ]), type="l", ylab="パラメータ", main="トピック8のタグの出現率のパラメータのサンプリング結果")
matplot(GAMMA, type="l", ylab="パラメータ", main="トピックと無関係のタグの出現率のパラメータのサンプリング結果")


##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(rbind(word_mu, phit)[, 1:50], 3)

#タグ出現率の事後推定量
tag_mu1 <- apply(OMEGA[, , burnin:(R/keep)], c(1, 2), mean)   #タグの出現率の事後平均
round(rbind(tag_mu1, omegat), 3)

#トピックと無関係のタグの事後推定量
round(rbind(colMeans(GAMMA[burnin:(R/keep), ]), gammat), 3) #無関係タグの事後平均