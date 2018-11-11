#####確率的潜在要素解析(PLCAモデル)#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(HMM)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(86751)

####データの発生####
k <- 10   #トピック数
d <- 2000   #文書数
v <- 500   #語彙数
f <- 300000   #要素数

##パラメータの設定
#ディレクリ分布のパラメータ
alpha01 <- rep(3.0, k)
alpha11 <- rep(0.25, k)
alpha12 <- rep(0.15, v)

#パラメータを生成
pi <- pit <- extraDistr::rdirichlet(1, alpha01)
theta0 <- extraDistr::rdirichlet(d, alpha11) * matrix(runif(d, 0.5, 2.5), nrow=d, ncol=k)
theta <- thetat <- t(theta0 / matrix(colSums(theta0), nrow=d, ncol=k, byrow=T))
phi <- phit <- extraDistr::rdirichlet(k, alpha12)
Mu <- array(0, dim=c(d, v, k))
Par <- matrix(0, nrow=k, ncol=d*v)


##モデルに基づきデータを生成
#要素ごとに多項分布からトピックを生成
Z <- rmnom(f, 1, pi)
z <- as.numeric(Z %*% 1:k) 

#トピックに基づき単語を生成
ID_d <- rep(0, f)
wd <- rep(0, f)

for(j in 1:k){
  print(j)
  index <- which(z==j)
  #文書の割当を生成
  Z1 <- rmnom(length(index), 1, theta[j, ])
  ID_d[index] <- as.numeric(Z1 %*% 1:d)
  
  #単語の割当を生成
  Z2 <- rmnom(length(index), 1, phi[j, ])
  wd[index] <- as.numeric(Z2 %*% 1:v)
}

#文書行列を作成
WX <- matrix(0, nrow=d, ncol=v)
for(i in 1:f){
  WX[ID_d[i], wd[i]] <- WX[ID_d[i], wd[i]] + 1
}
hist(rowSums(WX), xlab="単語数", main="単語頻度の分布", col="grey", breaks=20)

##インデックスを作成
doc_list <- list()
word_list <- list()
wd_list <- list()
for(i in 1:d){doc_list[[i]] <- which(ID_d==i)}
for(i in 1:v){word_list[[i]] <- which(wd==i)}
for(i in 1:v){wd_list[[i]] <- ID_d[word_list[[i]]]}


####マルコフ連鎖モンテカルロ法でPLCAを推定####
##アルゴリズムの設定
R <- 10000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##パラメータの真値
theta <- t(thetat)
phi <- phit
pi <- pit

##初期値を設定
theta0 <- t(extraDistr::rdirichlet(k, rowSums(WX)/sum(WX) * 200)) + 0.00001
theta <- theta0 / matrix(colSums(theta0), nrow=d, ncol=k)
phi0 <- extraDistr::rdirichlet(k, colSums(WX)/sum(WX) * 100) + 0.00001
phi <- phi0 / matrix(rowSums(phi0), nrow=k, ncol=v)
pi <- rep(1/k, k)

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 0.1 
beta01 <- 0.1
beta02 <- 1


##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
PI <- matrix(0, nrow=R/keep, ncol=k)
SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG) <- "integer"


#対数尤度の基準値
par1 <- rowSums(WX)/sum(WX)
par2 <- colSums(WX)/sum(WX)
LLst0 <- rep(0, f)
for(i in 1:v){
  n <- length(wd_list[[i]])
  LLst0[word_list[[i]]] <- par1[wd_list[[i]]] * par2[i]   #尤度
}
LLst <- sum(log(LLst0))


####ギブスサンプリングでHTMモデルのパラメータをサンプリング####
for(rp in 1:R){
  
  ##重み付き尤度からトピック分布を生成
  #PLCAの重み付き尤度とトピック割当確率を計算
  Li <- matrix(0, nrow=f, ncol=k)
  for(i in 1:v){
    n <- length(wd_list[[i]])
    r <- matrix(pi, nrow=n, ncol=k, byrow=T)
    Li[word_list[[i]], ] <- r * theta[wd_list[[i]], ] * matrix(phi[, i], nrow=n, ncol=k, byrow=T)   #重み付き尤度
  }
  topic_rate <- Li / rowSums(Li)   #トピック割当確率

  #多項分布からトピックを生成
  Zi <- rmnom(f, 1, topic_rate)
  z_vec <- as.numeric(Zi %*% 1:k)

  
  ##パラメータをサンプリング
  #混合率piをサンプリング
  psum <- colSums(Zi) + beta02
  pi <- extraDistr::rdirichlet(1, psum)
  
  #文書分布thetaをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi[doc_list[[i]], ])
  }
  wsum <- t(wsum0)+ beta01
  theta <- t(extraDistr::rdirichlet(k, wsum))
  
  #単語分布phiをサンプリング
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi[word_list[[j]], , drop=FALSE])
  }
  vf <- vf0 + alpha11
  phi <- extraDistr::rdirichlet(k, vf)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    PI[mkeep, ] <- pi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG <- SEG + Zi
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      print(c(sum(log(rowSums(Li))), LLst))
      print(round(cbind(theta[1:10, ], t(thetat[, 1:10])), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
      round(print(rbind(pi, pit)), 3)
    }
  }
}





