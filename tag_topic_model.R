#####タグ生成トピックモデル#####
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
#set.seed(423943)
#文書データの設定
k <- 8   #トピック数
d <- 1500   #文書数
v <- 300   #語彙数
w <- rpois(d, 160)   #1文書あたりの単語数
f <- sum(w)   #総単語数
a <- 75   #タグ数
x0 <- rpois(d, 15)
x <- ifelse(x0 < 1, 1, x0)
e <- sum(x)
 
#IDの設定
word_id <- rep(1:d, w)
aux_id <- rep(1:d, x)

#パラメータの設定
alpha0 <- rep(0.2, k)   #文書のディレクリ事前分布のパラメータ
alpha1 <- rep(0.25, a)   #タグのディレクリ事前分布のパラメーた
alpha2 <- rep(0.025, v)   #単語のディレクリ事前分布のパラメータ

#ディレクリ乱数の発生
thetat <- theta <- extraDistr::rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から発生
gammat <- gamma <- extraDistr::rdirichlet(k, alpha1)   #タグのトピック分布をディレクリ乱数から発生
phit <- phi <- extraDistr::rdirichlet(a, alpha2)   #単語のトピック分布をディレクリ乱数から発生


##多項分布からトピックおよび単語データを発生
WX <- matrix(0, nrow=d, ncol=v)
AX <- matrix(0, nrow=d, ncol=a)
z0 <- rep(0, sum(x)) 
Z1 <- list()
Z2 <- list()

#文書ごとにトピックと単語を逐次生成
for(i in 1:d){
  print(i)
  
  #文書のトピック分布を発生
  z <- rmnom(x[i], 1, theta[i, ])   #文書のトピック分布を発生
  
  #文書のトピック分布からタグを発生
  zd <- as.numeric(z %*% 1:k)   #0,1を数値に置き換える
  
  an <- rmnom(x[i], 1, gamma[zd, ])   #文書のトピックからタグを生成
  ad <- colSums(an)   #単語ごとに合計して1行にまとめる
  AX[i, ] <- ad
  
  #発生されたタグから単語を生成
  share <- rep(1:a, colSums(rmnom(w[i], 1, ad)))
  wn <- rmnom(w[i], 1, phi[share, ])   #多項分布から単語を生成
  wd <- colSums(wn)
  WX[i, ] <- wd
}

#データ行列を整数型行列に変更
storage.mode(WX) <- "integer"
storage.mode(AX) <- "integer"


####トピックモデル推定のためのデータと関数の準備####
##それぞれの文書中の単語の出現および補助情報の出現をベクトルに並べる
##データ推定用IDを作成
ID1_list <- list()
wd_list <- list()
ID2_list <- list()
ad_list <- list()

#求人ごとに求人IDおよび単語IDを作成
for(i in 1:nrow(WX)){
  print(i)
  
  #単語のIDベクトルを作成
  ID1_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
  W1 <- WX[i, (WX[i, ] > 0)]
  number <- rep(num2, W1)
  wd_list[[i]] <- number
  
  #補助情報のIDベクトルを作成
  ID2_list[[i]] <- rep(i, x[i])
  num1 <- (AX[i, ] > 0) * (1:a)
  num2 <- which(num1 > 0)
  A1 <- AX[i, (AX[i, ] > 0)]
  number <- rep(num2, A1)
  ad_list[[i]] <- number
}

#リストをベクトルに変換
ID1_d <- unlist(ID1_list)
ID2_d <- unlist(ID2_list)
wd <- unlist(wd_list)
ad <- unlist(ad_list)

##インデックスを作成
doc1_list <- list()
word_list <- list()
doc2_list <- list()
aux_list <- list()
for(i in 1:length(unique(ID1_d))) {doc1_list[[i]] <- which(ID1_d==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- which(wd==i)}
for(i in 1:length(unique(ID2_d))) {doc2_list[[i]] <- which(ID2_d==i)}
for(i in 1:length(unique(ad))) {aux_list[[i]] <- which(ad==i)}
gc(); gc()


####マルコフ連鎖モンテカルロ法で対応トピックモデルを推定####
##単語ごとに尤度と負担率を計算する関数
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
burnin <- 1000/keep

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- rep(1.0, k)
beta0 <- rep(0.5, v)
gamma0 <- rep(0.5, a)
alpha01m <- matrix(alpha01, nrow=d, ncol=k, byrow=T)
beta0m <- matrix(beta0, nrow=a, ncol=v)
gamma0m <- matrix(gamma0, nrow=k, ncol=a)
delta0m <- gamma0

##パラメータの初期値
#tfidfで初期値を設定
tf <- AX/rowSums(AX)
idf <- log(nrow(AX)/colSums(AX > 0))

theta <- rdirichlet(d, rep(1, k))   #文書トピックのパラメータの初期値
gamma <- rdirichlet(k, idf)   #タグトピックのパラメータの初期値
phi <- rdirichlet(a, rep(10, v))   #単語の出現率のパラメータの初期値

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(a, v, R/keep))
GAMMA <- array(0, dim=c(k, a, R/keep))
W_SEG <- matrix(0, nrow=f, ncol=a)
A_SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(W_SEG) <- "integer"
storage.mode(A_SEG) <- "integer"
gc(); gc()

##MCMC推定用配列
AXL <- AX[ID1_d, ]
tsum0 <- matrix(0, nrow=d, ncol=k)
vf0 <- matrix(0, nrow=k, ncol=a)
wf0 <- matrix(0, nrow=a, ncol=v)
af0 <- matrix(0, nrow=a, ncol=k)
aux_z <- rep(0, length(ad))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##生成したタグから単語をサンプリング
  #単語を生成する潜在タグをサンプリング
  word_bur <- t(phi)[wd, ] * AXL
  word_rate <- word_bur/rowSums(word_bur)
  Zi1 <- rmnom(f, 1, word_rate)
  Zi1[is.na(Zi1)] <- 0
  z_vec1 <- as.numeric(Zi1 %*% 1:a)
  
  #潜在タグから単語出現率をサンプリング
  for(j in 1:v){
    wf0[, j] <- colSums(Zi1[wd_list[[j]], ])
  }
  wf <- wf0 + beta0m
  phi <- extraDistr::rdirichlet(a, wf)

  
  ##タグトピックをサンプリング
  #タグごとにトピックの出現率を計算
  tag_rate <- burden_fr(theta, gamma, z_vec1, w, k)$Br

  #多項分布から単語トピックをサンプリング
  Zi2 <- rmnom(f, 1, tag_rate)   
  z_vec2 <- Zi2 %*% 1:k

  ##文書トピックのパラメータを更新
  #ディクレリ分布からthetaをサンプリング
  for(i in 1:d){
    tsum0[i, ] <- colSums(Zi2[doc1_list[[i]], ])
  }
  tsum <- tsum0 + alpha01m 
  theta <- extraDistr::rdirichlet(d, tsum)

  #ディクレリ分布からタグphiをサンプリング
  for(j in 1:a){
    vf0[, j] <- colSums(Zi2[z_vec1==j, , drop=FALSE])
  }
  vf <- vf0 + gamma0m
  gamma <- extraDistr::rdirichlet(k, vf)
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma

    #トピック割当はバーンイン期間を超えたら格納する
    if(rp >= burnin){
      A_SEG <- A_SEG + Zi2
      W_SEG <- W_SEG + Zi1
    }
    
    #サンプリング結果を確認
    print(rp)
    print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
    print(round(cbind(gamma[, 1:10], gammat[, 1:10]), 3))
    #print(round(cbind(phi[1:8, 1:10], phit[1:8, 1:10]), 3))
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

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, 1:10, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[2, 11:20, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[3, 21:30, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[4, 31:40, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(PHI[5, 41:50, ]), type="l", ylab="パラメータ", main="トピック5の単語の出現率のサンプリング結果")
matplot(t(PHI[6, 51:60, ]), type="l", ylab="パラメータ", main="トピック6の単語の出現率のサンプリング結果")
matplot(t(PHI[7, 61:70, ]), type="l", ylab="パラメータ", main="トピック7の単語の出現率のサンプリング結果")
matplot(t(PHI[8, 71:80, ]), type="l", ylab="パラメータ", main="トピック8の単語の出現率のサンプリング結果")

#タグの出現確率のサンプリング結果
matplot(t(OMEGA[1, 1:10, ]), type="l", ylab="パラメータ", main="トピック1のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[2, 6:15, ]), type="l", ylab="パラメータ", main="トピック2のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[3, 16:25, ]), type="l", ylab="パラメータ", main="トピック3のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[4, 21:30, ]), type="l", ylab="パラメータ", main="トピック4のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[5, 26:35, ]), type="l", ylab="パラメータ", main="トピック5のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[6, 31:40, ]), type="l", ylab="パラメータ", main="トピック6のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[7, 36:45, ]), type="l", ylab="パラメータ", main="トピック7のタグの出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[8, 41:50, ]), type="l", ylab="パラメータ", main="トピック8のタグの出現率のパラメータのサンプリング結果")
matplot(GAMMA[, 41:50], type="l", ylab="パラメータ", main="トピックと無関係のタグの出現率のパラメータのサンプリング結果1")
matplot(GAMMA[, 51:60], type="l", ylab="パラメータ", main="トピックと無関係のタグの出現率のパラメータのサンプリング結果2")

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
round(rbind(colMeans(GAMMA[burnin:(R/keep), ]), gammat), 3)   #無関係タグの事後平均


