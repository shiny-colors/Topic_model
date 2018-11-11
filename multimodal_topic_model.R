#####マルチモーダルトピックモデル#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
detach("package:gtools", unload=TRUE)
detach("package:bayesm", unload=TRUE)
library(extraDistr)
library(monomvn)
library(glmnet)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(54876)

####データの発生####
#set.seed(423943)
#データの設定
m <- 3   #文書データ数
k <- 8   #トピック数
d <- 2500   #文書数

#各文書データの設定
#語彙数の設定
v1 <- 250   
v2 <- 200
v3 <- 150

#1文書あたりの単語数
w1 <- rpois(d, 150)  
w2 <- rpois(d, 125)
w3 <- rpois(d, 100)

#パラメータの設定
alpha0 <- rep(0.5, k)   #全体のディレクリ事前分布のパラメータ
alpha1 <- rep(0.25, v1)   #文書1の単語のディレクリ事前分布のパラメータ
alpha2 <- rep(0.3, v2)   #文書2の単語のディクレリ事前分布のパラメータ
alpha3 <- rep(0.3, v3)   #文書3の単語のディクレリ事前分布のパラメータ

#ディレクリ乱数の発生
thetat <- theta <- rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から発生
phit1 <- phi1 <- rdirichlet(k, alpha1)   #単語のトピック分布をディレクリ乱数から発生
phit2 <- phi2 <- rdirichlet(k, alpha2)   #単語のトピック分布をディレクリ乱数から発生
phit3 <- phi3 <- rdirichlet(k, alpha3)   #単語のトピック分布をディレクリ乱数から発生


#多項分布の乱数からデータを発生
WX1 <- matrix(0, nrow=d, ncol=v1)
WX2 <- matrix(0, nrow=d, ncol=v2)
WX3 <- matrix(0, nrow=d, ncol=v3)
Z1 <- list()
Z2 <- list()
Z3 <- list()

for(i in 1:d){
  print(i)
  
  #文書1のトピック分布を発生
  z1 <- t(rmultinom(w1[i], 1, theta[i, ]))   #文書のトピック分布を発生
  
  #文書1のトピック分布から単語を発生
  zn <- z1 %*% c(1:k)   #0,1を数値に置き換える
  zdn <- cbind(zn, z1)   #apply関数で使えるように行列にしておく
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi1[x[1], ])))   #文書のトピックから単語を生成
  wdn <- colSums(wn)   #単語ごとに合計して1行にまとめる
  WX1[i, ] <- wdn  
  Z1[[i]] <- zdn[, 1]   #文書トピックを格納
  
  
  #文書2のトピック分布を発生
  z2 <- t(rmultinom(w2[i], 1, theta[i, ]))   #文書のトピック分布を発生
  
  #文書2のトピック分布から単語を発生
  zn <- z2 %*% c(1:k)   #0,1を数値に置き換える
  zdn <- cbind(zn, z2)   #apply関数で使えるように行列にしておく
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi2[x[1], ])))   #文書のトピックから単語を生成
  wdn <- colSums(wn)   #単語ごとに合計して1行にまとめる
  WX2[i, ] <- wdn 
  Z2[[i]] <- zdn[, 1]   #文書トピックを格納
  
  
  #文書3のトピック分布を発生
  z3 <- t(rmultinom(w3[i], 1, theta[i, ]))   #文書のトピック分布を発生
  
  #文書3のトピック分布から単語を発生
  zn <- z3 %*% c(1:k)   #0,1を数値に置き換える
  zdn <- cbind(zn, z3)   #apply関数で使えるように行列にしておく
  wn <- t(apply(zdn, 1, function(x) rmultinom(1, 1, phi3[x[1], ])))   #文書のトピックから単語を生成
  wdn <- colSums(wn)   #単語ごとに合計して1行にまとめる
  WX3[i, ] <- wdn 
  Z3[[i]] <- zdn[, 1]   #文書トピックを格納
}

#データ行列を整数型行列に変更
storage.mode(WX1) <- "integer"
storage.mode(WX2) <- "integer"
storage.mode(WX3) <- "integer"

####マルチモーダルLDA推定のためのデータと関数の準備####
##それぞれの文書中の単語の出現および補助情報の出現をベクトルに並べる
##データ推定用IDを作成
ID1_list <- list()
ID2_list <- list()
ID3_list <- list()
wd1_list <- list()
wd2_list <- list()
wd3_list <- list()

#求人ごとに求人IDおよび単語IDを作成
for(i in 1:d){
  print(i)
  
  #文書1の単語のIDベクトルを作成
  ID1_list[[i]] <- rep(i, w1[i])
  num1 <- (WX1[i, ] > 0) * (1:v1)
  num2 <- subset(num1, num1 > 0)
  W1 <- WX1[i, (WX1[i, ] > 0)]
  number <- rep(num2, W1)
  wd1_list[[i]] <- number
  
  #文書2の単語のIDベクトルを作成
  ID2_list[[i]] <- rep(i, w2[i])
  num1 <- (WX2[i, ] > 0) * (1:v2)
  num2 <- subset(num1, num1 > 0)
  W2 <- WX2[i, (WX2[i, ] > 0)]
  number <- rep(num2, W2)
  wd2_list[[i]] <- number
  
  #文書3の単語のIDベクトルを作成
  ID3_list[[i]] <- rep(i, w3[i])
  num1 <- (WX3[i, ] > 0) * (1:v3)
  num2 <- subset(num1, num1 > 0)
  W3 <- WX3[i, (WX3[i, ] > 0)]
  number <- rep(num2, W3)
  wd3_list[[i]] <- number
}

#リストをベクトルに変換
ID1_d <- unlist(ID1_list)
ID2_d <- unlist(ID2_list)
ID3_d <- unlist(ID3_list)
wd1 <- unlist(wd1_list)
wd2 <- unlist(wd2_list)
wd3 <- unlist(wd3_list)
storage.mode(ID1_d) <- "integer"
storage.mode(ID2_d) <- "integer"
storage.mode(ID3_d) <- "integer"
storage.mode(wd1) <- "integer"
storage.mode(wd2) <- "integer"
storage.mode(wd3) <- "integer"


##インデックスを作成
doc1_list <- list()
doc2_list <- list()
doc3_list <- list()
word1_list <- list()
word2_list <- list()
word3_list <- list()
for(i in 1:length(unique(ID1_d))) {doc1_list[[i]] <- which(ID1_d==i)}
for(i in 1:length(unique(wd1))) {word1_list[[i]] <- which(wd1==i)}
for(i in 1:length(unique(ID2_d))) {doc2_list[[i]] <- which(ID2_d==i)}
for(i in 1:length(unique(wd2))) {word2_list[[i]] <- which(wd2==i)}
for(i in 1:length(unique(ID3_d))) {doc3_list[[i]] <- which(ID3_d==i)}
for(i in 1:length(unique(wd3))) {word3_list[[i]] <- which(wd3==i)}
gc(); gc()


####マルコフ連鎖モンテカルロ法でマルチモーダルLDAを推定####
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
keep <- 2
burnin <- 1000/keep
iter <- 0

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- rep(1, k)
beta01 <- rep(0.5, v1)
beta02 <- rep(0.5, v2)
beta03 <- rep(0.5, v3)
alpha01m <- matrix(alpha01, nrow=d, ncol=k, byrow=T)
beta01m <- matrix(beta01, nrow=v1, ncol=k)
beta02m <- matrix(beta02, nrow=v2, ncol=k)
beta03m <- matrix(beta03, nrow=v3, ncol=k)

##パラメータの初期値を設定
theta.ini <- runif(k, 0.5, 2)
phi1.ini <- runif(v1, 0.5, 1)
phi2.ini <- runif(v2, 0.5, 1)
phi3.ini <- runif(v3, 0.5, 1)
theta <- rdirichlet(d, theta.ini)   #全体の文書トピックのパラメータの初期値
phi1 <- rdirichlet(k, phi1.ini)   #文書1の単語トピックのパラメータの初期値
phi2 <- rdirichlet(k, phi2.ini)   #文書2の単語トピックのパラメータの初期値
phi3 <- rdirichlet(k, phi3.ini)   #文書3の単語トピックのパラメータの初期値


##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI1 <- array(0, dim=c(k, v1, R/keep))
PHI2 <- array(0, dim=c(k, v2, R/keep))
PHI3 <- array(0, dim=c(k, v3, R/keep))
W1_SEG <- matrix(0, nrow=sum(w1), ncol=k)
W2_SEG <- matrix(0, nrow=sum(w2), ncol=k)
W3_SEG <- matrix(0, nrow=sum(w3), ncol=k)
storage.mode(W1_SEG) <- "integer"
storage.mode(W2_SEG) <- "integer"
storage.mode(W3_SEG) <- "integer"
gc(); gc()

##MCMC推定用配列
vec <- 1/1:k
vf01 <- matrix(0, nrow=v1, ncol=k)
vf02 <- matrix(0, nrow=v2, ncol=k)
vf03 <- matrix(0, nrow=v3, ncol=k)
wsum1 <- wsum2 <- wsum3 <- matrix(0, nrow=d, ncol=k)


####マルコフ連鎖モンテカルロ法でパラメータをサンプリング####
for(rp in 1:R){
  
  ##文書1のトピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_rate1 <- burden_fr(theta, phi1, wd1, w1, k)$Br
  
  #多項分布から単語トピックをサンプリング
  word_cumsums <- rowCumsums(word_rate1)
  rand <- matrix(runif(nrow(word_rate1)), nrow=nrow(word_rate1), ncol=k)   #一様乱数
  Zi1 <- ((k+1) - (word_cumsums > rand) %*% rep(1, k)) %*% vec   #トピックをサンプリング
  Zi1[Zi1!=1] <- 0
  
  ##文書1の単語トピックのパラメータを更新
  #ディクレリ分布からphiをサンプリング
  for(i in 1:v1){
    vf01[i, ] <- colSums(Zi1[word1_list[[i]], ])
  }
  vf <- t(vf01 + beta01m)   #ディクレリ分布のパラメータ
  phi1 <- extraDistr::rdirichlet(k, vf)   #ディクレリ分布からphiをサンプリング
  
  
  ##文書2のトピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_rate2 <- burden_fr(theta, phi2, wd2, w2, k)$Br
  
  #多項分布から単語トピックをサンプリング
  word_cumsums <- rowCumsums(word_rate2)
  rand <- matrix(runif(nrow(word_rate2)), nrow=nrow(word_rate2), ncol=k)   #一様乱数
  Zi2 <- ((k+1) - (word_cumsums > rand) %*% rep(1, k)) %*% vec   #トピックをサンプリング
  Zi2[Zi2!=1] <- 0
  
  ##文書2の単語トピックのパラメータを更新
  #ディクレリ分布からphiをサンプリング
  for(i in 1:v2){
    vf02[i, ] <- colSums(Zi2[word2_list[[i]], ])
  }
  vf <- t(vf02 + beta02m)   #ディクレリ分布のパラメータ
  phi2 <- extraDistr::rdirichlet(k, vf)   #ディクレリ分布からphiをサンプリング
  
  
  ##文書3のトピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_rate3 <- burden_fr(theta, phi3, wd3, w3, k)$Br
  
  #多項分布から単語トピックをサンプリング
  word_cumsums <- rowCumsums(word_rate3)
  rand <- matrix(runif(nrow(word_rate3)), nrow=nrow(word_rate3), ncol=k)   #一様乱数
  Zi3 <- ((k+1) - (word_cumsums > rand) %*% rep(1, k)) %*% vec   #トピックをサンプリング
  Zi3[Zi3!=1] <- 0
  
  ##文書3の単語トピックのパラメータを更新
  #ディクレリ分布からphiをサンプリング
  for(i in 1:v3){
    vf03[i, ] <- colSums(Zi3[word3_list[[i]], ])
  }
  vf <- t(vf03 + beta03m)   #ディクレリ分布のパラメータ
  phi3 <- extraDistr::rdirichlet(k, vf)   #ディクレリ分布からphiをサンプリング
  
  
  ##発生させたトピックから共通のthetaをサンプリング
  #ディクレリ分布のパラメータを計算
  for(i in 1:d){
    wsum1[i, ] <- colSums(Zi1[doc1_list[[i]], ])
    wsum2[i, ] <- colSums(Zi2[doc2_list[[i]], ])
    wsum3[i, ] <- colSums(Zi3[doc3_list[[i]], ])
  }
  wsum <- wsum1 + wsum2 + wsum3 + alpha01m   #ディクレリ分布のパラメータ
  theta <- extraDistr::rdirichlet(d, wsum)   #ディクレリ分布からトピック割当をサンプリング
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi2
    PHI3[, , mkeep] <- phi3
    
    if(rp >= burnin){
      W1_SEG <- W1_SEG + Zi1
      W2_SEG <- W2_SEG + Zi2
      W3_SEG <- W3_SEG + Zi3
    }
    
    #サンプリング結果を確認
    print(rp)
    print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
    print(round(cbind(phi1[, 1:10], phit1[, 1:10]), 3))
  }
}

####サンプリング結果の推定値と要約####
##バーンイン期間の設定
burnin <- 1000/keep
RS <- R/keep
z_range <- length(burnin:(R/keep))

##サンプリング結果の可視化
matplot(t(THETA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ推定値")
matplot(t(THETA[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ推定値")
matplot(t(THETA[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ推定値")
matplot(t(THETA[2000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ推定値")
matplot(t(PHI1[1, 1:5, ]), type="l", xlab="サンプリング回数", ylab="パラメータ推定値")
matplot(t(PHI2[2, 10:15, ]), type="l", xlab="サンプリング回数", ylab="パラメータ推定値")
matplot(t(PHI3[3, 20:25, ]), type="l", xlab="サンプリング回数", ylab="パラメータ推定値")

##サンプリング結果の要約
#文書中の単語ごとのトピック割当確率
round(W1_SEG / rowSums(W1_SEG), 3)
round(W2_SEG / rowSums(W2_SEG), 3)
round(W3_SEG / rowSums(W3_SEG), 3)

#文書のトピック割当確率
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)

#単語のトピック割当確率
round(cbind(t(apply(PHI1[, , burnin:RS], c(1, 2), mean)), t(phit1)), 3)
round(cbind(t(apply(PHI2[, , burnin:RS], c(1, 2), mean)), t(phit2)), 3)
round(cbind(t(apply(PHI3[, , burnin:RS], c(1, 2), mean)), t(phit3)), 3)



