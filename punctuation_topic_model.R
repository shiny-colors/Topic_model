#####Punctuationトピックモデル#####
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

#set.seed(8079)

####データの発生####
#文書データの設定
k <- 10   #トピック数
d <- 1000   #文書数
a <- rpois(d, rgamma(d, 22, 2.5))   #文章数
a[a < 5] <- ceiling(runif(sum(a < 3), 5, 15))
s <- sum(a)   #総文章数
v <- 300   #語彙数
w <- rpois(s, 21)   #1文章あたりの単語数
w[w==0] <- 1
f <- sum(w)   #総単語数

##IDの設定
id_a <- rep(1:s, w)
id_u <- rep(1:d, a)
id_d <- c()
id_w <- c()
id_s <- c()

for(i in 1:d){
  freq <- w[rep(1:d, a)==i]
  id_d <- c(id_d, rep(i, sum(freq)))
  id_w <- c(id_w, 1:sum(freq))
  id_s <- c(id_s, 1:a[i])
}

##パラメータの設定
#ディリクレ事前分布の設定
alpha1 <- rep(0.2, k)   #文書トピックのディリクレ事前分布のパラメータ
alpha2 <- rep(0.3, v)   #単語のディリクレ事前分布のパラメータ

#ディリクレ乱数の発生
thetat <- theta <- extraDistr::rdirichlet(d, alpha1)   #文書ごとのトピック分布
phit <- phi <- extraDistr::rdirichlet(k, alpha2)   #単語ごとのトピック分布

##多項分布からトピックおよび単語データを発生
WX <- matrix(0, nrow=d, ncol=v)
AX <- matrix(0, nrow=s, ncol=v)
Z0 <- list()

for(i in 1:d){
  print(i)

  #文書トピック分布を発生
  z <- rmnom(a[i], 1, theta[i, ])
  zd <- as.numeric(z %*% 1:k)
  
  #文書のトピックから単語を生成
  an <- rmnom(sum(id_u==i), w[id_u==i], phi[zd, ])
  AX[id_u==i, ] <- an
  WX[i, ] <- colSums(an)
  Z0[[i]] <- z
}

#データ行列を変換
Z <- do.call(rbind, Z0)
storage.mode(WX) <- "integer"
storage.mode(AX) <- "integer"


####トピックモデル推定のためのデータと関数の準備####
##データ推定用IDの作成
ID1_list <- list()
ID2_list <- list()
wd_list <- list()

#文書ごとに文書IDおよび単語IDを作成
for(i in 1:nrow(AX)){
  print(i)
  
  #単語のIDベクトルを作成
  ID1_list[[i]] <- rep(id_u[i], w[i])
  ID2_list[[i]] <- rep(id_s[i], w[i])

  num1 <- (AX[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
  A1 <- AX[i, (AX[i, ] > 0)]
  number <- rep(num2, A1)
  wd_list[[i]] <- number
}

#リストをベクトルに変換
ID1_d <- unlist(ID1_list)
ID2_d <- unlist(ID2_list)
wd <- unlist(wd_list)

##インデックスを作成
doc1_list <- list()
doc2_list <- list()
word_list <- list()
for(i in 1:length(unique(ID1_d))) {doc1_list[[i]] <- which(ID1_d==i)}
for(i in 1:length(unique(id_a))) {doc2_list[[i]] <- which(id_a==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- which(wd==i)}
gc(); gc()


####マルコフ連鎖モンテカルロ法でPunctuationトピックモデルを推定####
##文章ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #負担係数の格納用
  for(j in 1:k){
    #負担係数を計算
    Bi <- rep(theta[, j], w) * phi[j, wd]   #尤度
    Bur[, j] <- Bi   
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
disp <- 10
burnin <- 1000/keep

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01m <- matrix(1.0, nrow=d, ncol=k, byrow=T)
beta0m <- matrix(0.5, nrow=k, ncol=v)

##パラメータの初期値の設定
theta <- rdirichlet(d, rep(10, k))   #文書トピックのパラメータの初期値
phi <- rdirichlet(k, colSums(WX)/100)   #単語の出現率のパラメータの初期値

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
SEG <- matrix(0, nrow=s, ncol=k)
storage.mode(SEG) <- "integer"
gc(); gc()

##MCMC推定用の配列
wf0 <- matrix(0, nrow=d, ncol=k)
vf0 <- matrix(0, nrow=k, ncol=v)


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){

  ##文章ごとにトピックをサンプリング
  #文章レベルでの尤度と負担率を計算
  LLi <- AX %*% t(log(phi))   #文章ごとの対数尤度
  LLi_max <- apply(LLi, 1, max)   
  Bur <- theta[id_u, ] * exp(LLi - LLi_max)   #文章レベルの尤度
  Br <- Bur / rowSums(Bur)   #負担率
  
  #多項分布から文章トピックをサンプリング
  Zi <- rmnom(s, 1, Br)

  #トピック分布のパラメータをサンプリング
  Zi_word <- Zi[id_a, ]
  for(i in 1:d){
    wf0[i, ] <- colSums(Zi_word[doc1_list[[i]], ])
  }
  wf <- wf0 + alpha01m
  theta <- extraDistr::rdirichlet(d, wf)
  
  
  ##トピックごとに単語の出現率をサンプリング
  for(j in 1:v){
    vf0[, j] <- colSums(Zi_word[word_list[[j]], ])
  }
  vf <- vf0 + beta0m
  phi <- extraDistr::rdirichlet(k, vf)
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp >= burnin){
      SEG <- SEG + Zi
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
    }
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep
RS <- R/keep

##サンプリング結果の可視化
matplot(t(THETA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[500, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 1, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 100, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 300, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

##サンプリング結果の要約統計量
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phi)), 3)
cbind(round(SEG / sum(SEG[1, ]), 3), Z)

