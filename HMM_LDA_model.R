#####HMM-LDAモデル#####
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
'%!in%' <- function(a,b) ! a %in% b

#set.seed(2506787)

####データの発生####
##データの設定
k1 <- 8   #syntax数
k2 <- 15   #トピック数
d <- 3000   #文書数
v1 <- 700   #トピックに関係のある語彙数
v2 <- 500   #トピック以外のsyntaxに関係のある語彙数
v <- v1 + v2   #総語彙数
w <- rpois(d, rgamma(d, 40, 0.15))   #文書あたりの単語数
f <- sum(w)   #総単語数
vec_k1 <- rep(1, k1)
vec_k2 <- rep(1, k2)

##IDの設定
d_id <- rep(1:d, w)
t_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))

##パラメータを設定
#ディレクリ分布のパラメータ
alpha01 <- seq(2.5, 0.5, length=k1)
alpha02 <- matrix(0.5, nrow=k1, ncol=k1); alpha02[-k1, k1] <- 8.0; alpha02[k1, k1] <- 3.0
alpha11 <- rep(0.15, k2)
alpha21 <- c(rep(0.05, v1), rep(0.00025, v2))

#syntaxの事前分布
alloc <- as.numeric(rmnom(v2, 1, extraDistr::rdirichlet(k1-1, rep(5.0, k1-1))) %*% 1:(k1-1))
alpha22 <- cbind(matrix(0.0001, nrow=k1-1, ncol=v1), matrix(0.0025, nrow=k1-1, ncol=v2))
for(j in 1:(k1-1)){
  index <- which(alloc==j) + v1
  alpha22[j, index] <- 2.5
}

##モデルに基づき単語を生成
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  #ディレクリ分布よりパラメータを生成
  pi1 <- pit1 <- as.numeric(extraDistr::rdirichlet(1, alpha01))
  pi2 <- pit2 <- extraDistr::rdirichlet(k1, alpha02)
  theta <- thetat <- extraDistr::rdirichlet(d, alpha11)
  phi <- extraDistr::rdirichlet(k2, alpha21)
  psi <- extraDistr::rdirichlet(k1-1, alpha22)
  
  #単語出現確率が低いトピックを入れ替える
  index1 <- which(colMaxs(psi) < (k1*10)/f); index1 <- index1[index1 > v1]
  index2 <- which(colMaxs(phi) < (k2*10)/f); index2 <- index2[index2 <= v1]
  for(j in 1:length(index1)){
    psi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1.0, k1-1))) %*% 1:(k1-1)), index1[j]] <- (k1*10)/f
  }
  for(j in 1:length(index2)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1.0, k2))) %*% 1:k2), index2[j]] <- (k2*10)/f
  }
  psit <- psi
  phit <- phi
  
  ##HMM-LDAモデルに基づき単語を生成
  wd_list <- Z1_list <- Z2_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    z1_vec <- rep(0, w[i])
    z2_vec <- rep(0, w[i])
    words <- matrix(0, nrow=w[i], ncol=v)
    
    for(j in 1:w[i]){
      if(j==1){
        #文書の先頭単語のsyntaxを生成
        z1 <- rmnom(1, 1, pi1)
        z1_vec[j] <- as.numeric(z1 %*% 1:k1)
      } else {
        #先頭以降はマルコフ推移に基づきsyntaxを生成
        z1 <- rmnom(1, 1, pi2[z1_vec[j-1], ])
        z1_vec[j] <- as.numeric(z1 %*% 1:k1)
      }
    }
    sum(z1_vec==k1)
    
    #トピック分布を生成
    index_topic <- which(z1_vec==k1)
    z2 <- rmnom(length(index_topic), 1, theta[i, ])
    z2_vec[index_topic] <- as.numeric(z2 %*% 1:k2)
    
    #トピック分布に基づき単語を生成
    words[-index_topic, ] <- rmnom(w[i]-length(index_topic), 1, psi[z1_vec[-index_topic], ])   #syntaxに関連する単語
    words[index_topic, ] <- rmnom(length(index_topic), 1, phi[z2_vec[index_topic], ])   #トピックに関連する単語
    

    #データを格納
    wd_list[[i]] <- as.numeric(words %*% 1:v)
    WX[i, ] <- colSums(words) 
    Z1_list[[i]] <- z1_vec
    Z2_list[[i]] <- z2_vec
  }
  if(min(colSums(WX)) > 0){
    break
  }
}

#リスト形式をベクトル形式に変換
z1 <- unlist(Z1_list); Z1 <- matrix(as.numeric(table(1:f, z1)), nrow=f, ncol=k1)
z2 <- unlist(Z2_list)
wd <- unlist(wd_list)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))   #単語ベクトルを行列化
word_dt <- t(word_data)

#インデックスを作成
d_list <- word_list <- list()
for(i in 1:d){d_list[[i]] <- which(d_id==i)}
for(i in 1:v){word_list[[i]] <- which(wd==i)}


####マルコフ連鎖モンテカルロ法でHMM-LDAモデルを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / as.numeric(Bur %*% vec_k)   #負担率
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}


####ギブスサンプラーでHMM-LDAモデルを推定####
##アルゴリズムの設定
R <- 3000
keep <- 2  
iter <- 0
burnin <- 500
disp <- 10
er <- 0.00005

##インデックスとデータの設定
#データの設定
d_data <- sparseMatrix(1:f, d_id, x=rep(1, f), dims=c(f, d))   #文書ベクトルを行列化
d_dt <- t(d_data)

#先頭と後尾のインデックスを作成
max_word <- max(t_id)
index_t11 <- which(t_id==1)
index_t12 <- rep(0, d)
for(i in 1:d){
  index_t12[i] <- max(d_list[[i]])
}

#中間のインデックスを作成
index_list_t21 <- index_list_t22 <- list()
for(j in 2:max_word){
  index_list_t21[[j]] <- which(t_id==j)-1
  index_list_t22[[j]] <- which(t_id==j)
}
index_t21 <- sort(unlist(index_list_t21))
index_t22 <- sort(unlist(index_list_t22))


##事前分布の設定
alpha01 <- 0.01 
alpha02 <- 0.01
beta01 <- 0.01
beta02 <- 0.01

##パラメータの真値
#パラメータの初期値
pi1 <- as.numeric(pit1)
pi2 <- pit2
theta <- thetat
phi <- phit
psi <- psit

#HMMの潜在変数の初期値
z1_vec <- z1
Zi1 <- matrix(as.numeric(table(1:f, z1)), nrow=f, ncol=k1) %*% 1:k1

##初期値を設定
#パラメータの初期値
pi1 <- as.numeric(extraDistr::rdirichlet(1, rep(2.0, k1)))
pi2 <- extraDistr::rdirichlet(k1, rep(2.0, k1))
theta <- extraDistr::rdirichlet(d, rep(2.0, k2))
phi <- extraDistr::rdirichlet(k2, rep(1.0, v))
psi <- extraDistr::rdirichlet(k1-1, rep(1.0, v))

#HMMの潜在変数の初期値
Zi1 <- rmnom(f, 1, rep(1/k1, k1))
z1_vec <- as.numeric(Zi1 %*% 1:k1)


##パラメータの格納用配列
PI1 <- matrix(0, nrow=R/keep, ncol=k1)
PI2 <- array(0, dim=c(k1, k1, R/keep))
THETA <- array(0, dim=c(d, k2, R/keep))
PHI <- array(0, dim=c(k2, v, R/keep))
PSI <- array(0, dim=c(k1-1, v, R/keep))
SEG1 <- matrix(0, nrow=f, ncol=k1)
SEG2 <- matrix(0, nrow=f, ncol=k2)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##対数尤度の基準値
#ユニグラムモデルの対数尤度
LLst <- sum(WX %*% log(colSums(WX)/sum(WX)))   

#ベストな対数尤度
LL <- c()
LL1 <- sum(log(as.numeric((Z1[z1!=k1, -k1] * t(psit)[wd[z1!=k1], ]) %*% rep(1, k1-1))))
LL2 <- sum(log(as.numeric(((thetat[d_id, ] * t(phit)[wd, ])[z1==k1, ]) %*% vec_k2)))
LLbest <- LL1 + LL2


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとの尤度と混合率を設定
  #syntaxとトピックモデルの尤度
  Li01 <- t(psi)[wd, ]   #syntaxごとの尤度
  word_par <- burden_fr(theta, phi, wd, d_id, k2, vec_k2) 
  Li02 <- rowSums(word_par$Bur)   #トピックモデルの期待尤度
  Li0 <- cbind(Li01, Li02)   #尤度の結合
  
  #HMMの混合率
  pi_dt1 <- pi_dt2 <- matrix(1, nrow=f, ncol=k1)
  pi_dt1[index_t11, ] <- matrix(pi1, nrow=d, ncol=k1, byrow=T)   #文書の先頭と後尾の混合率
  pi_dt1[index_t22, ] <- pi2[z1_vec[index_t21], ]   #1単語前の混合率
  pi_dt2[index_t21, ] <- t(pi2)[z1_vec[index_t22], ]   #1単語後の混合率
  

  ##多項分布からHMMの潜在変数をサンプリング
  #潜在変数の割当確率
  Li <- pi_dt1 * pi_dt2 * Li0   #結合分布
  z1_rate <- Li / as.numeric(Li %*% vec_k1)   #割当確率
  
  #潜在変数をサンプリング
  Zi1 <- rmnom(f, 1, z1_rate)
  z1_vec <- as.numeric(Zi1 %*% 1:k1)
  n1 <- sum(Zi1[, 1:(k1-1)]); n2 <- sum(Zi1[, k1])
  index_topic <- which(Zi1[, k1]==1)
  
  ##HMMのパラメータをサンプリング
  #ディリクレ分布から推移確率をサンプリング
  rf11 <- colSums(Zi1[index_t11, ]) + alpha01
  rf12 <- t(Zi1[index_t21, ]) %*% Zi1[index_t22, ] + alpha02
  pi1 <- as.numeric(extraDistr::rdirichlet(1, rf11))
  pi2 <- extraDistr::rdirichlet(k1, rf12)

  
  #ディクレリ分布からsyntaxのパラメータをサンプリング
  Zi1_syntax <- Zi1[-index_topic, ]
  df <- t(as.matrix(word_dt[, -index_topic] %*% Zi1_syntax)) + alpha02
  psi <- extraDistr::rdirichlet(k1-1, df)
  
  
  ##単語のトピックをサンプリング
  Zi2 <- matrix(0, nrow=f, ncol=k2)
  Zi2[index_topic, ] <- rmnom(n2, 1, word_par$Br[index_topic, ])   #多項分布よりトピックをサンプリング
  z2_vec <- as.numeric(Zi2 %*% 1:k2)

  
  ##トピックモデルのパラメータをサンプリング
  #トピック分布をサンプリング
  Zi2_topic <- Zi2[index_topic, ]
  wsum <- as.matrix(d_dt[, index_topic] %*% Zi2_topic) + beta01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #単語分布をサンプリング
  vf <- t(as.matrix(word_dt[, index_topic] %*% Zi2_topic)) + beta02
  phi <- extraDistr::rdirichlet(k2, vf)
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    PI1[mkeep, ] <- pi1
    PI2[, , mkeep] <- pi2
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    PSI[, , mkeep] <- psi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    if(rp%%disp==0){
      #対数尤度を計算
      LL1 <- sum(log(as.numeric((Zi1[-index_topic, -k1] * t(psi)[wd[-index_topic], ]) %*% rep(1, k1-1))))
      LL2 <- sum(log(as.numeric(((theta[d_id, ] * t(phi)[wd, ])[index_topic, ]) %*% vec_k2)))
      LL <- c(LL, LL1 + LL2)
      
      #サンプリング結果を表示
      print(rp)
      print(c(LL1+LL2, LLbest, LLst))
      print(round(cbind(pi2, pit2), 3))
      print(round(cbind(psi[, (v1-4):(v1+5)], psit[, (v1-4):(v1+5)]), 3))
    }
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA[100, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA[1000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA[2000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#単語の出現確率のサンプリング結果
matplot(t(PHI[, 1, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[, 200, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[, 400, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[, 500, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(PSI[, 1, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PSI[, 200, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PSI[, 400, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PSI[, 500, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")


##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu1 <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
word1 <- round(t(rbind(word_mu1, phit)), 3)
word_mu2 <- apply(OMEGA[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
word2 <- round(t(rbind(word_mu2, omegat)), 3)
word <- round(t(rbind(word_mu1, word_mu2, phit, omegat)), 3)
colnames(word) <- 1:ncol(word)

word_mu3 <- apply(GAMMA[burnin:(R/keep), ], 2, mean)   #単語の出現率の事後平均
round(rbind(word_mu3, gamma=gammat), 3)


##トピックの事後分布の要約
round(seg1_mu <- SEG1 / rowSums(SEG1), 3)
round(seg2_mu <- SEG2 / rowSums(SEG2), 3)
