#####Twitter LDA#####
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

####データの発生####
##文書データの設定
hh <- 2000   #ユーザー数
tweet <- rpois(hh, rgamma(hh, 20.0, 0.2))
d <- sum(tweet)
w <- rpois(d, 12.5)
f <- sum(w)
v1 <- 700   #トピックの語彙数
v2 <- 500   #一般語の語彙数
v <- v1+v2   #総語彙数
k <- 15   #トピック数

#IDの設定
u_id <- rep(1:hh, tweet)   
t_id <- as.numeric(unlist(tapply(1:d, u_id, rank)))
index_id <- list()
for(i in 1:hh){
  index_id[[i]] <- which(u_id==i)
}


##パラメータの設定
#ディクレリ事前分布を設定
alpha1 <- rep(0.15, k)   #ユーザー固有のディクレリ分布のパラメータ
alpha21 <- c(rep(0.04, v1), rep(0.0001, v2))   #トピック語のディクレリ分布のパラメータ
alpha22 <- c(rep(0.01, v1), rep(5.0, v2))   #一般語のディクレリ分布の事前分布のパラメータ
beta <- c(4.5, 3.5)   #一般語かどうかのベータ分布のパラメータ


##すべての単語が出現するまでデータの生成を続ける
rp <- 0
repeat {
  rp <- rp + 1
  
  #ディクレリ分布からパラメータを生成
  thetat <- theta <- extraDistr::rdirichlet(hh, alpha1)   #ユーザートピックの生成
  lambda <- lambdat <- rbeta(hh, beta[1], beta[2])   #一般語とトピック語の比率
  phi <- extraDistr::rdirichlet(k, alpha21)   #トピック語の出現率の生成
  gamma <- gammat <- extraDistr::rdirichlet(1, alpha22)   #一般語の出現率の生成
  
  #単語出現確率が低いトピックを入れ替える
  index <- which(colMaxs(phi) < (k*10)/f & alpha21==max(alpha21))
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, alpha1)) %*% 1:k), index[j]] <- (k*10)/f
  }
  phit <- phi
  
  ##多項分布からトピックおよび単語データを生成
  WX <- matrix(0, nrow=d, ncol=v)
  Z_list <- y_list <- word_list <- list()
  index_word1 <- 1:v1
  index_word2 <- (v1+1):v
  
  #tweetごとに1つのトピックを割り当て単語を生成
  for(i in 1:hh){
    
    #tweetごとにトピックを生成
    z <- rmnom(tweet[i], 1, theta[i, ])
    z_vec <-as.numeric(z %*% 1:k)
    index_hh <- index_id[[i]]
    
    #tweetに割り当てられたトピックから単語を生成
    for(j in 1:nrow(z)){
      
      #トピックに関係あるかどうかの潜在変数
      freq <- w[index_hh[j]]
      y <- rbinom(freq, 1, lambda[i])
      index_y1 <- which(y==1); index_y0 <- which(y==0)
      
      #潜在変数に基づいてtweetの単語を生成
      word <- matrix(0, nrow=freq, ncol=v)
      word[index_y1, ] <- rmnom(sum(y), 1, phi[z_vec[j], ])   #トピック語の生成
      word[index_y0, ] <- rmnom(sum(1-y), 1, gamma)   #一般語の生成
      
      #生成したデータを格納
      word_list[[index_hh[j]]] <- as.numeric(word %*% 1:v)
      WX[index_hh[j], ] <- colSums(word)
      y_list[[index_hh[j]]] <- y
    }
    #トピックを格納
    Z_list[[i]] <- as.numeric(z_vec)
  }
  #break条件
  print(min(colSums(WX)))
  if(min(colSums(WX)) > 0){
    break
  }
}

#リストを変換
wd <- unlist(word_list)
y_vec <- unlist(y_list)
z_vec <- unlist(Z_list)
storage.mode(WX) <- "integer"

#スパースデータを作成
sparse_data <- as(WX, "CsparseMatrix")
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))


####マルコフ連鎖モンテカルロ法で対応トピックモデルを推定####
##アイテムごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / as.numeric(Bur %*% vec_k)   #負担率
  r <- colSums(Br) / sum(Br)   #混合率
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##アルゴリズムの設定
R <- 3000
keep <- 2  
iter <- 0
burnin <- 500
disp <- 10

##データとインデックスの設定
#データの設定
user_id <- rep(u_id, w)
d_id <- rep(1:d, w)
vec_k <- rep(1, k)

#インデックスの設定
user_dt <- sparseMatrix(user_id, 1:f, x=rep(1, f), dims=c(hh, f))
u_dt <- sparseMatrix(u_id, 1:d, x=rep(1, d), dims=c(hh, d))
d_dt <- sparseMatrix(d_id, 1:f, x=rep(1, f), dims=c(d, f))
wd_dt <- t(word_data)
user_n <- rowSums(user_dt)


##事前分布の設定
#ハイパーパラメータの事前分布
alpha1 <- 0.1
beta1 <- 0.01  
gamma1 <- 0.01 
s0 <- 1
v0 <- 1

##パラメータの真値
theta <- thetat
lambda <- lambdat
phi <- phit 
gamma <- as.numeric(gammat)


##パラメータの初期値
theta <- extraDistr::rdirichlet(d, rep(2.0, k))
lambda <- rbeta(hh, 2.0, 2.0)
phi <- extraDistr::rdirichlet(k, rep(2.0, v))
gamma <- as.numeric(extraDistr::rdirichlet(1, rep(2.0, v)))


##パラメータの格納用配列
THETA <- array(0, dim=c(hh, k, R/keep))
LAMBDA <- matrix(0, nrow=R/keep, ncol=hh)
PHI <- array(0, dim=c(k, v, R/keep))
GAMMA <- matrix(0, nrow=R/keep, ncol=v)
SEG <- matrix(0, nrow=d, ncol=k)
storage.mode(SEG) <- "integer"


##対数尤度の基準値
#ユニグラムモデルの対数尤度
LLst <- sum(word_data %*% log(colSums(word_data) / f))

#ベストな対数尤度
LL_topic <- sum(log(rowSums(thetat[user_id, ] * t(phit)[wd, ])[y_vec==1]))
LL_general <- sum(log((gammat[wd])[y_vec==0]))
LLbest <- LL_topic + LL_general


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##トピック語か一般語かどうかをサンプリング
  #トピックの尤度
  Lho_topic <- theta[user_id, ] * t(phi)[wd, ]
  Lho_general <- gamma[wd]

  #トピックと一般語の割当確率
  r <- lambda[user_id]
  Lho <- cbind(r * as.numeric(Lho_topic %*% vec_k), (1-r) * Lho_general)   #トピックと一般語の期待尤度
  allocation_rate <- Lho / rowSums(Lho)
  
  #二項分布から割当をサンプリング
  y <- rbinom(f, 1, allocation_rate)
  
  #ベータ分布から混合率をサンプリング
  n <- as.numeric(user_dt %*% y)
  s1 <- n + s0
  v1 <- user_n - n + v0
  lambda <- rbeta(hh, s1, v1)   #パラメータをサンプリング

  ##ツイート単位でトピックをサンプリング
  #トピックの割当確率を設定
  Lho <- as.matrix(d_dt %*% (Lho_topic * y))   #一般語を除いたトピック語の尤度
  topic_rate <- Lho / as.numeric(Lho %*% vec_k)
  index_na <- which(is.na(as.numeric(topic_rate %*% vec_k)))
  
  #多項分布からトピックをサンプリング
  Zi <- matrix(0, nrow=d, ncol=k)
  Zi[-index_na, ] <- rmnom(d-length(index_na), 1, topic_rate[-index_na, ])
  
  
  ##ディリクレ分布からパラメータをサンプリング
  #トピック分布をサンプリング
  wsum <- as.matrix(u_dt %*% Zi) + alpha1
  theta <- extraDistr::rdirichlet(hh, wsum)
  
  #トピック語分布をサンプリング
  vsum <- as.matrix(t(wd_dt %*% (Zi[d_id, ] * y))) + beta1
  phi <- extraDistr::rdirichlet(k, vsum)

  #一般語分布をサンプリング
  gsum <- as.numeric(wd_dt %*% (1-y)) + beta1
  gamma <- as.numeric(extraDistr::rdirichlet(1, gsum))
  

  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    LAMBDA[mkeep, ] <- lambda
    PHI[, , mkeep] <- phi
    GAMMA[mkeep, ] <- gamma
    if(burnin > rp){
      SEG <- SEG + Zi
    }
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL <- sum(log(rowSums(allocation_rate * cbind(as.numeric(Lho_topic %*% vec_k), Lho_general))))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(cbind(phi[, 696:705], phit[, 696:705]), 3))
    print(round(rbind(gamma[691:710], gammat[691:710]), 3))
  }
}


####サンプリング結果の可視化と要約####
burnin <- 2000/keep
RS <- R/keep

##サンプリング結果をプロット
matplot(t(THETA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 1, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 700, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 701, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(GAMMA[, 691:700], type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(GAMMA[, 701:710], type="l", xlab="サンプリング回数", ylab="パラメータ")

##パラメータのサンプリング結果の要約
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)   #ユーザーのトピック割当確率
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phit)), 3)   #トピック語のトピック別の出現確率
round(cbind(colMeans(GAMMA[burnin:RS, ]), as.numeric(gammat)), 3)   #一般語の出現確率

##トピックのサンプリング結果の要約
round(cbind(SEG/rowSums(SEG), z_vec), 3)   #tweetごとのトピック割当の要約

