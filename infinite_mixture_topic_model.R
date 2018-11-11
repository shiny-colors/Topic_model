#####無限混合トピックモデル#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
detach("package:gtools", unload=TRUE)
library(extraDistr)
library(monomvn)
library(glmnet)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

####データの発生####
#set.seed(423943)
#データの設定
k <- 10   #トピック数
d <- 2000   #文書数
v <- 250   #語彙数
w <- rpois(d, 150)   #1文書あたりの単語数

#パラメータの設定
alpha0 <- round(runif(k, 0.1, 1.25), 3)   #文書のディレクリ事前分布のパラメータ
alpha1 <- rep(0.25, v)   #単語のディレクリ事前分布のパラメータ

#ディレクリ乱数の発生
theta0 <- theta <- rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から発生
phi0 <- phi <- rdirichlet(k, alpha1)   #単語のトピック分布をディレクリ乱数から発生

#多項分布の乱数からデータを発生
WX <- matrix(0, nrow=d, ncol=v)
Z <- list()
for(i in 1:d){
  z <- t(rmultinom(w[i], 1, theta[i, ]))   #文書のトピック分布を発生
  zn <- z %*% c(1:k)   #0,1を数値に置き換える
  zdn <- cbind(zn, z)   #apply関数で使えるように行列にしておく
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi[x[1], ])))   #文書のトピックから単語を生成
  wdn <- colSums(wn)   #単語ごとに合計して1行にまとめる
  WX[i, ] <- wdn  
  Z[[i]] <- zdn[, 1]
  print(i)
}

####EMアルゴリズムでトピックモデルを推定####
####トピックモデルのためのデータと関数の準備####
##それぞれの文書中の単語の出現をベクトルに並べる
##データ推定用IDを作成
ID_list <- list()
wd_list <- list()

#求人ごとに求人IDおよび単語IDを作成
for(i in 1:nrow(WX)){
  print(i)
  ID_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * c(1:v) 
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
for(i in 1:length(unique(ID_d))) {doc_list[[i]] <- which(ID_d==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- which(wd==i)}
gc(); gc()


####マルコフ連鎖モンテカルロ法で無限混合トピックモデルを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #負担係数の格納用
  for(kk in 1:k){
    #負担係数を計算
    Bi <- rep(theta[, kk], w) * (phi[kk, c(wd)])   #尤度
    Bur[, kk] <- Bi   
  }
  Br <- Bur / rowSums(Bur)   #負担率の計算
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}


##アルゴリズムの設定
R <- 10000
keep <- 4
rbeta <- 1.5
iter <- 0
k0 <- 2   #初期トピック数

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- rep(1.0, k0)
beta0 <- rep(0.5, v)
alpha01m <- matrix(alpha01, nrow=d, ncol=k0, byrow=T)
beta0m <- matrix(beta0, nrow=v, ncol=k0)

#集中度パラメータ
tau1 <- 1
tau2 <- 2

##パラメータの初期値
theta.ini <- runif(k0, 0.5, 2)
phi.ini <- runif(v, 0.5, 1)
theta <- rdirichlet(d, theta.ini)   #文書トピックのパラメータの初期値
phi <- rdirichlet(k0, phi.ini)   #単語トピックのパラメータの初期値

#単語ごとにトピックの出現率を計算
word_rate <- burden_fr(theta, phi, wd, w, k0)$Br   #既存トピックの出現率

#多項分布から単語トピックをサンプリング
vec <- 1/(1:k0)
word_cumsums <- rowCumsums(word_rate)
rand <- matrix(runif(nrow(word_rate)), nrow=nrow(word_rate), ncol=k0)   #一様乱数
Zi1 <- ((k0+1) - (word_cumsums > rand) %*% rep(1, k0)) %*% vec   #トピックをサンプリング
Zi1[Zi1!=1] <- 0

#テーブル数を更新
Zi1[, 1]


##混合数の更新
r0 <- c(colSums(Zi1), tau2)
pi0 <- extraDistr::rdirichlet(1, r0)

##文書トピックのパラメータを更新
#ディクレリ分布からthetaをサンプリング
for(i in 1:d){
  wsum0[i, ] <- colSums(Zi1[doc_list[[i]], ]) 
}
theta_prior <- alpha01m * matrix(pi0[, 1:k0], nrow=d, ncol=k0, byrow=T)
wsum <- cbind(wsum0 + theta_prior, tau1 * pi0[, k0+1])   #ディクレリ分布のパラメータ
theta_tau <- extraDistr::rdirichlet(d, wsum)[, k0+1]   #ディクレリ分布からthetaをサンプリング
theta_tau


##パラメータの格納用配列
max_k <- 20
THETA <- array(0, dim=c(d, max_k, R/keep))
PHI <- array(0, dim=c(max_k, v, R/keep))
W_SEG <- matrix(0, nrow=sum(w), ncol=max_k)
storage.mode(W_SEG) <- "integer"
gc(); gc()

##MCMC推定用配列
wsum0 <- matrix(0, nrow=d, ncol=k0)
vf0 <- matrix(0, nrow=v, ncol=k0)
vec <- 1/1:k0

####ギブスサンプリングでトピックモデルのパラメータをサンプリング####
for(rp in 1:R){
  #トピックするを更新
  k0 <- ncol(theta) 
  k1 <- k0 + 1
  
  ##単語トピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_rate_old <- burden_fr(theta, phi, wd, w, k0)$Bur   #既存トピックの出現率
  word_rate_new <- pi0 * theta_tau[ID_d]   #新規トピックの出現率
  word_rate <- cbind(word_rate_old, word_rate_new)
  
  #多項分布から単語トピックをサンプリング
  word_cumsums <- rowCumsums(word_rate)
  rand <- matrix(runif(nrow(word_rate)), nrow=nrow(word_rate), ncol=k1)   #一様乱数
  Zi1 <- ((k1+1) - (word_cumsums > rand) %*% rep(1, k1)) %*% vec   #トピックをサンプリング
  Zi1[Zi1!=1] <- 0
  
  #Zi1 <- rmnom(nrow(word_rate), 1, word_rate)

  
  ##文書トピックのパラメータを更新
  #ディクレリ分布からthetaをサンプリング
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi1[doc_list[[i]], ]) 
  }
  wsum <- wsum0 + alpha01m   #ディクレリ分布のパラメータ
  theta <- extraDistr::rdirichlet(d, wsum)   #ディクレリ分布からthetaをサンプリング
  
  ##単語トピックのパラメータを更新
  #ディクレリ分布からphiをサンプリング
  for(i in 1:v){
    vf0[i, ] <- colSums(Zi1[word_list[[i]], ])
  }
  vf <- t(vf0 + beta0m)   #ディクレリ分布のパラメータ
  phi <- extraDistr::rdirichlet(k, vf)   #ディクレリ分布からphiをサンプリング
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    mkeep1 <- rp/keep
    THETA[, , mkeep1] <- theta
    PHI[, , mkeep1] <- phi
    #W_SEG[mkeep2, ] <- word_z

    #サンプリング結果を確認
    print(rp)
    print(round(cbind(theta[1:10, ], theta0[1:10, ]), 3))
    #print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))

  }
}



