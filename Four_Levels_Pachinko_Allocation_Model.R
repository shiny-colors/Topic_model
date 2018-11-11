#####Four Levels Pachinko Allocation Model#####
options(warn=0)
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

#set.seed(93441)
####データの発生####
##データの設定
L <- 2
k1 <- 4   #上位トピック数
k2 <- 15   #下位トピック数
d <- 5000   #文書数
v <- 1200   #語彙数
w <- rpois(d, rgamma(d, 50, 0.3))   #文書あたりの単語数
f <- sum(w)   #総単語数
vec_k1 <- rep(1, k1)
vec_k2 <- rep(1, k2)

#文書IDの設定
d_id <- rep(1:d, w)
a_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))

##パラメータの設定
#ディレクリ分布のパラメータを設定
alpha1 <- rep(0.2, k1)
alpha2 <- rep(0.15, k2)
beta1 <- rep(0.2, k2)
beta2 <- rep(0.05, v)

##モデルに基づきデータを生成
rp <- 0
repeat { 
  rp <- rp + 1
  print(rp)
  
  #ディレクリ分布からパラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(d, alpha1)
  theta2 <- thetat2 <- array(0, dim=c(d, k2, k1))
  for(j in 1:k1){
    theta2[, , j] <- thetat2[, , j] <- extraDistr::rdirichlet(d, alpha2)
  }
  gamma <- gammat <- extraDistr::rdirichlet(k1, beta1)
  phi <- extraDistr::rdirichlet(k2, beta2)
  
  #単語出現確率が低いトピックを入れ替える
  index <- which(colMaxs(phi) < (k2*10)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(2.0, k2))) %*% 1:k2), index[j]] <- (k2*10)/f
  }
  phit <- phi
  
  ##文書ごとにトピックと単語を生成
  Z1_list <- list()
  Z2_list <- list()
  word_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    #上位トピックを生成
    z1 <- rmnom(w[i], 1, theta1[i, ])
    z1_vec <- as.numeric(z1 %*% 1:k1)
    
    #下位トピックを生成
    z2 <- rmnom(w[i], 1, t(theta2[i, , z1_vec]))
    z2_vec <- as.numeric(z2 %*% 1:k2)
    
    #単語を生成
    word <- rmnom(w[i], 1, phi[z2_vec, ])
    
    #データを格納
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
    word_list[[i]] <- as.numeric(word %*% 1:v)
    WX[i, ] <- colSums(word)
  }
  if(min(colSums(WX)) > 0){
    break
  }
}

#データを変換
Z1 <- do.call(rbind, Z1_list)
Z2 <- do.call(rbind, Z2_list)
storage.mode(WX) <- "integer"
wd <- unlist(word_list)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
word_data_T <- t(word_data)
rm(word_list); rm(Z1_list); rm(Z2_list)
gc(); gc()


####マルコフ連鎖モンテカルロ法でPAMを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / as.numeric(Bur %*% vec_k)   #負担率
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}

##アルゴリズムの設定
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##インデックスの設定
d_data <- sparseMatrix(1:f, d_id, x=rep(1, f), dims=c(f, d))
d_data_T <- t(d_data)

##事前分布の設定
alpha1 <- 0.5
alpha2 <- 0.25
beta1 <- 0.1

##パラメータの真値
theta1 <- thetat1
theta2 <- thetat2
phi <- phit
Zi1 <- Z1 
Zi2 <- Z2

##初期値の設定
theta1 <- extraDistr::rdirichlet(d, rep(2.0, k1))
theta2 <- array(0, dim=c(d, k2, k1))
for(j in 1:k1){
  theta2[, , j] <- extraDistr::rdirichlet(d, rep(2.0, k2))
}
phi <- extraDistr::rdirichlet(k2, rep(2.0, v))
Zi1 <- rmnom(f, 1, theta1[d_id, ])
z1_vec <- as.numeric(Z1 %*% 1:k1)

##パラメータの格納用配列
THETA1 <- array(0, dim=c(d, k1, R/keep))
THETA2 <- array(0, dim=c(d, k2, k1, R/keep))
PHI <- array(0, dim=c(k2, v, R/keep))
SEG1 <- matrix(0, nrow=f, ncol=k1)
SEG2 <- matrix(0, nrow=f, ncol=k2)

##対数尤度の基準値
#ユニグラムモデルの対数尤度
LLst <- sum(word_data %*% log(colSums(word_data) / f))

#ベストモデルの対数尤度
theta_topic <- thetat2[d_id, , ]
theta_k2 <- matrix(0, nrow=f, ncol=k2)
for(j in 1:k1){
  theta_k2 <- theta_k2 + theta_topic[, , j] * Z1[, j]
}
LLbest <- sum(log(as.numeric((theta_k2 * t(phit)[wd, ]) %*% vec_k2)))   #対数尤度


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){

  ##下位トピックをサンプリング
  #上位トピック割当に基づき下位トピック分布を設定
  theta_topic <- theta2[d_id, , ]
  theta_k2 <- matrix(0, nrow=f, ncol=k2)
  for(j in 1:k1){
    theta_k2 <- theta_k2 + theta_topic[, , j] * Zi1[, j]
  }
  
  #トピック分布の尤度と負担率を推定
  Lho2 <- theta_k2 * t(phi)[wd, ]   #トピック割当ごとの尤度
  topic_rate <- Lho2 / as.numeric(Lho2 %*% vec_k2)   #負担率
  
  #多項分布よりトピックをサンプリング
  Zi2 <- rmnom(f, 1, topic_rate)
  z2_vec <- as.numeric(Zi2 %*% 1:k2)
  
  
  ##上位トピックをサンプリング
  #下位トピック割当に基づき上位のトピック出現分布を設定
  theta_k1 <- matrix(0, nrow=f, ncol=k1)
  for(j in 1:k2){
    theta_k1 <- theta_k1 + theta_topic[, j, ] * Zi2[, j] 
  }
  
  #トピック分布の尤度と負担率を推定
  Lho1 <- theta1[d_id, ] * theta_k1   #トピック割当ごとの尤度
  topic_rate <- Lho1 / as.numeric(Lho1 %*% vec_k1)   #負担率
  
  #多項分布よりトピックをサンプリング
  Zi1 <- rmnom(f, 1, topic_rate)
  z1_vec <- as.numeric(Zi1 %*% 1:k1)
  
  
  ##パラメータをサンプリング
  #上位トピックのパラメータをサンプリング
  wsum1 <- as.matrix(d_data_T %*% Zi1) + alpha1   #ディリクレ分布のパラメータ
  theta1 <- extraDistr::rdirichlet(d, wsum1)   #トピック分布をサンプリング
  
  #上位トピックごとに下位トピックのパラメータをサンプリング
  for(j in 1:k1){
    wsum2 <- as.matrix(d_data_T %*% (Zi1[, j] * Zi2)) + alpha2     #ディクレリ分布のパラメータ
    theta2[, , j] <- extraDistr::rdirichlet(d, wsum2)   #トピック分布をサンプリング
  }
  
  ##単語分布のパラメータをサンプリング
  #ディクレリ分布のパラメータ
  vsum <- as.matrix(t(word_data_T %*% Zi2)) + beta1
  phi <- extraDistr::rdirichlet(k2, vsum)   #ディクレリ分布から単語分布をサンプリング
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , , mkeep] <- theta2
    PHI[, , mkeep] <- phi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL <- sum(log(as.numeric((theta_k2 * t(phi)[wd, ]) %*% vec_k2)))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(cbind(theta1[1:5, ], thetat1[1:5, ]), 3))
    print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep
RS <- R/keep

##サンプリング結果の可視化
#上位トピックのトピック分布のサンプリング結果
matplot(t(THETA1[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="上位トピックのサンプリング結果")
matplot(t(THETA1[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="上位トピックのサンプリング結果")
matplot(t(THETA1[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="上位トピックのサンプリング結果")
matplot(t(THETA1[15, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="上位トピックのサンプリング結果")
matplot(t(THETA1[20, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="上位トピックのサンプリング結果")

#下位トピックのトピック分布のサンプリング結果
matplot(t(THETA2[1, , 1, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")
matplot(t(THETA2[1, , 2, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")
matplot(t(THETA2[1, , 3, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")
matplot(t(THETA2[1, , 4, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")
matplot(t(THETA2[10, , 1, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")
matplot(t(THETA2[10, , 2, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")
matplot(t(THETA2[10, , 3, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")
matplot(t(THETA2[10, , 4, ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位トピックのサンプリング結果")

#単語分布のサンプリング結果
matplot(t(PHI[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="単語分布のサンプリング結果")
matplot(t(PHI[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="単語分布のサンプリング結果")
matplot(t(PHI[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="単語分布のサンプリング結果")
matplot(t(PHI[15, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="単語分布のサンプリング結果")

##事後分布の要約統計量
#トピック分布の事後平均
topic_mu1 <- apply(THETA1[, , burnin:RS], c(1, 2), mean)
topic_mu2 <- array(0, dim=c(d, k2, k1))
for(j in 1:k1){
  topic_mu2[, , j] <- apply(THETA2[, , j, burnin:RS], c(1, 2), mean)
}
round(topic_mu1, 3)
round(topic_mu2[, , 1], 3)

#単語分布の事後平均
round(phi_mu <- t(apply(PHI[, , burnin:RS], c(1, 2), mean)), 3)
round(cbind(phi_mu, t(phit)), 3)

#トピック割当の事後分布
topic_rate1 <- SEG1 / rowSums(SEG1) 
topic_allocation1 <- apply(topic_rate1, 1, which.max)
round(data.frame(真値=Z1 %*% 1:k1, 推定=topic_allocation1, z=topic_rate1), 3)

topic_rate2 <- SEG2 / rowSums(SEG2)
topic_allocation2 <- apply(topic_rate2, 1, which.max)
round(data.frame(真値=Z2 %*% 1:k2, 推定=topic_allocation2, z=topic_rate2), 3)
