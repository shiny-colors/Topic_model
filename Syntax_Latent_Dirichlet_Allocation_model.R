#####Syntax Latent Dirichlet Allocationモデル#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(extraDistr)
library(reshape2)
library(plyr)
library(dplyr)
library(ggplot2)


#set.seed(21437)

####データの発生####
#set.seed(423943)
#データの設定
k <- 10   #トピック数
d <- 2000   #文書数
v1 <- 300   #トピックに関係のある語彙数
v2 <- 100   #トピックに関係のない語彙数
v <- v1 + v2   #総語彙数
w <- rpois(d, rgamma(d, 60, 0.50))   #1文書あたりの単語数
f <- sum(w)


#パラメータの設定
alpha0 <- rep(0.15, k)   #文書のディレクリ事前分布のパラメータ
alpha1 <- c(rep(0.4, v1), rep(0.0075, v2))   #トピックに関係のある単語のディレクリ事前分布のパラメータ
alpha2 <- c(rep(0.1, v1), rep(10, v2))   #一般語のディレクリ事前分布のパラメータ

#ディレクリ乱数の発生
thetat <- theta <- extraDistr::rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から発生
phit <- phi <- extraDistr::rdirichlet(k, alpha1)   #単語のトピック分布をディレクリ乱数から発生
gammat <- gamma <- extraDistr::rdirichlet(1, alpha2)   #一般語の単語分布をディレクリ乱数から発生
betat <- beta <- rbeta(d, 20, 15)


#多項分布の乱数からデータを発生
WX <- matrix(0, nrow=d, ncol=v)
y_list <- list()
Z_list <- list()

for(i in 1:d){
  #文書のトピックを生成
  z <- rmnom(w[i], 1, theta[i, ])   #文書のトピック分布を発生
  z_vec <- as.numeric(z %*% c(1:k))   #トピック割当をベクトル化
  
  #一般語かどうかを生成
  y0 <- rbinom(w[i], 1, beta)
  index <- which(y0==1)
  
  #トピックから単語を生成
  wn1 <- rmnom(length(index), 1, phi[z_vec[index], ])   #文書のトピックから単語を生成
  wn2 <- rmnom(1, w[i]-length(index), gamma)   #一般語を生成
  wdn <- colSums(wn1) + colSums(wn2)   #単語ごとに合計して1行にまとめる
  WX[i, ] <- wdn
  Z_list[[i]] <- z
  y_list[[i]] <- y0
  print(i)
}

#リスト形式を変換
Z <- do.call(rbind, Z_list)
y_vec <- unlist(y_list)


####トピックモデル推定のためのデータと関数の準備####
##データ推定用IDを作成
ID_list <- list()
wd_list <- list()

#求人ごとに求人IDおよび単語IDを作成
for(i in 1:nrow(WX)){
  print(i)
  
  #単語のIDベクトルを作成
  ID_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
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
disp <- 20
burnin <- 1000/keep

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 1.0
beta01 <- 0.5
beta02 <- 0.5
beta03 <- c(f/50, f/50)


##パラメータの初期値
#tfidfで初期値を設定
tf <- WX/rowSums(WX)
idf1 <- log(nrow(WX)/colSums(WX > 0))
idf2 <- log(nrow(WX)/colSums(WX==0))

theta <- extraDistr::rdirichlet(d, rep(1, k))   #文書トピックのパラメータの初期値
phi <- extraDistr::rdirichlet(k, idf1*10)   #単語トピックのパラメータの初期値
gamma <- extraDistr::rdirichlet(1, idf2*100)   #一般語のパラメータの初期値
r <- 0.5   #混合率の初期値

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
GAMMA <- matrix(0, nrow=R/keep, v)
SEG <- matrix(0, nrow=f, ncol=k)
Y <- rep(0, f)
storage.mode(SEG) <- "integer"
gc(); gc()

##MCMC推定用配列
wsum0 <- matrix(0, nrow=d, ncol=k)
vf0 <- matrix(0, nrow=k, ncol=v)
df0 <- rep(0, v)


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語トピックをサンプリング
  #トピックの出現率と尤度を推定
  par <- burden_fr(theta, phi, wd, w, k)
  LH1 <- par$Bur
  word_rate <- par$Br
  
  #トピックに関係のの出現率を推定
  LH2 <- gamma[wd]
  
  
  ##一般語かどうかをサンプリング
  Bur1 <- r * rowSums(LH1)
  Bur2 <- (1-r) * LH2
  switch_rate <- Bur1 / (Bur1 + Bur2)
  y <- rbinom(f, 1, switch_rate) 
  index_y <- which(y==1)
  
  #ベータ分布から混合率を更新
  par <- sum(y)
  r <- rbeta(1, par+beta03[1], f-par+beta03[2])
  
  ##多項分布から単語トピックをサンプリング
  Zi <- rmnom(f, 1, word_rate)   
  Zi[-index_y, ] <- 0
  z_vec <- as.numeric(Zi %*% 1:k)
  
  
  ##パラメータをサンプリング
  #トピック分布をディクレリ分布からサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #トピック語の分布をディクレリ分布からサンプリング
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi[word_list[[j]], ])
  }
  vf <- vf0 + beta01
  phi <- extraDistr::rdirichlet(k, vf)
  
  #一般語の分布をディクレリ分布からサンプリング
  y0 <- 1-y
  df0 <- rep(0, v)
  for(j in 1:v){
    df0[j] <- sum(y0[word_list[[j]]])
  }
  df <- df0 + beta02
  gamma <- extraDistr::rdirichlet(1, df)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    GAMMA[mkeep, ] <- gamma 
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      SEG <- SEG + Zi
      Y <- Y + y
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      print(c(mean(y), mean(y_vec)))
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(phi[, 296:305], phit[, 296:305]), 3))
      print(round(rbind(gamma[296:305], gammat[296:305]), 3))
    }
  }
}


####サンプリング結果の可視化と要約####
burnin <- 1000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA[1, , ]), type="l", ylab="パラメータ", main="文書1のトピック分布のサンプリング結果")
matplot(t(THETA[2, , ]), type="l", ylab="パラメータ", main="文書2のトピック分布のサンプリング結果")
matplot(t(THETA[3, , ]), type="l", ylab="パラメータ", main="文書3のトピック分布のサンプリング結果")
matplot(t(THETA[4, , ]), type="l", ylab="パラメータ", main="文書4のトピック分布のサンプリング結果")

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, 296:305, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[2, 296:305, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[3, 296:305, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[4, 296:305, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")

#一般語の出現確率のサンプリング結果
matplot(t(PHI[1, 286:295, ]), type="l", ylab="パラメータ", main="単語の出現率のサンプリング結果")
matplot(t(PHI[2, 296:305, ]), type="l", ylab="パラメータ", main="単語の出現率のサンプリング結果")
matplot(t(PHI[3, 306:315, ]), type="l", ylab="パラメータ", main="単語の出現率のサンプリング結果")


##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu1 <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(rbind(word_mu1, phit)[, 276:325], 3)

word_mu2 <- apply(GAMMA[burnin:(R/keep), ], 2, mean)   #単語の出現率の事後平均
round(rbind(word_mu, gamma=gammat), 3)

