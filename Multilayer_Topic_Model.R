#####Multilayer Topic Model#####
options(warn=2)
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
#データの設定
k0 <- 5   #上位階層のトピック数
k1 <- 10   #下位階層のトピック数
k_len <- k0
d <- 3000   #文書数
v <- 700   #語彙数
w <- rpois(d, rgamma(d, 100, 0.55))
f <- sum(w)

#文書IDの設定
d_id <- rep(1:d, w)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:w[i])}

#語彙トピックのインデックス
for(j in 1:100){
  v_index <- c(1, round(sort(runif(k_len-1, 100, v-100))), v)
  x <- v_index[2:length(v_index)] - v_index[1:(length(v_index)-1)]
  if(min(x) > 100) break
}

##パラメータの設定
#ディリクレ分布のパラメータ
alpha01 <- rep(0.25, k0)
alpha11 <- rep(0.2, k1)

beta01 <- matrix(0.3, nrow=k0, ncol=k_len)
diag(beta01) <- 2.0
beta02 <- list()
for(j in 1:k_len){
  beta <- rep(0.0001, v)
  beta[v_index[j]:v_index[j+1]] <- 0.1
  beta02[[j]] <- beta
}

#パラメータを生成
theta01 <- thetat01 <- extraDistr::rdirichlet(d, alpha01)
phi01 <- phit01 <- extraDistr::rdirichlet(k0, beta01)
theta02 <- thetat02 <- extraDistr::rdirichlet(d, alpha11)
phi02 <- phit02 <- list()
for(j in 1:k_len){
  par <- extraDistr::rdirichlet(k1, beta02[[j]]) + runif(v, 10^-100, 10^-50)
  phi02[[j]] <- phit02[[j]] <- par / rowSums(par)
}


##階層トピックモデルからデータを生成
Z1_list <- list()
Z2_list <- list()
S_list <- list()
word_list <- list()
WX_list <- list()

for(i in 1:d){
  ##上位階層のデータを生成
  #上位階層のトピックを生成
  z1 <- rmnom(w[i], 1, theta01[i, ])
  z1_vec <- as.numeric(z1 %*% 1:k0)
  
  #下位階層の割当を生成
  s <- rmnom(w[i], 1, phi01[z1_vec, ])
  s_vec <- as.numeric(s %*% 1:k_len)
  
  ##下位階層のデータを生成
  z2_vec <- c()
  words <- matrix(0, nrow=w[i], ncol=v)
  for(j in 1:w[i]){
    #下位階層のトピックを生成
    z2 <- rmnom(1, 1, theta02[i,])
    z2_vec <- c(z2_vec, as.numeric(z2 %*% 1:k1))

    #トピックから単語を生成
    words[j, ] <- rmnom(1, 1, phi02[[s_vec[j]]][z2_vec[j], ])
  }
  
  ##データを格納
  Z1_list[[i]] <- z1_vec
  Z2_list[[i]] <- z2_vec
  S_list[[i]] <- s_vec
  word_list[[i]] <- words
  WX_list[[i]] <- colSums(words)
}

##リストを変換
Z1 <- unlist(Z1_list)
Z2 <- unlist(Z2_list)
s <- unlist(S_list); s_freq <- table(s)
Data <- do.call(rbind, word_list)
wd <- as.numeric(Data %*% 1:v)
WX <- do.call(rbind, WX_list)
storage.mode(WX) <- "integer"
sparse_data <- as(Data, "CsparseMatrix")
rm(Data)
hist(colSums(WX), xlab="頻度", main="単語の頻度分布", col="grey", breaks=25)
summary(colSums(WX))

##インデックスを作成
doc_list <- list()
wd_list <- list()
for(i in 1:d){doc_list[[i]] <- which(d_id==i)}
for(i in 1:v){wd_list[[i]] <- which(wd==i)}


####マルコフ連鎖モンテカルロ法で階層トピックモデルを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #負担係数の格納用
  for(j in 1:k){
    #負担係数を計算
    Bi <- rep(theta[, j], w) * phi[j, wd]   #尤度
    Bur[, j] <- Bi   
  }
  Br <- Bur / rowSums(Bur)   #負担率の計算
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}


##アルゴリズムの設定
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##パラメータの事前分布
alpha01 <- 10
alpha02 <- 1
beta01 <- 0.1
beta02 <- 0.0001

##パラメータの真値
theta1 <- thetat01
theta2 <- thetat02
phi1 <- phit01
phi2 <- phit02
r <- phi1[Z1, ]

##パラメータの初期値
#ディリクレ分布のパラメータ
beta1 <- matrix(0.3, nrow=k0, ncol=k_len)
diag(beta1) <- 1.0
beta2 <- list()
for(j in 1:k_len){ 
  beta <- rep(0.1, v)
  beta[v_index[j]:v_index[j+1]] <- 2.0
  beta2[[j]] <- beta
}

#パラメータを生成
theta1 <- extraDistr::rdirichlet(d, rep(0.25, k0))
theta2 <- extraDistr::rdirichlet(d, alpha11 <- rep(0.2, k1))
phi1 <- extraDistr::rdirichlet(k0, beta1)
phi2 <- list()
for(j in 1:k_len){
  par <- extraDistr::rdirichlet(k1, beta2[[j]]) + runif(v, 10^-100, 10^-50)
  phi2[[j]] <- par / rowSums(par)
}
r <- matrix(1/k_len, nrow=f, ncol=k_len)


##パラメータの格納用配列
THETA1 <- array(0, dim=c(d, k0, R/keep))
THETA2 <- array(0, dim=c(d, k1, R/keep))
PHI1 <- array(0, dim=c(k0, k_len, R/keep))
PHI2 <- array(0, dim=c(k_len*k1, v, R/keep))
S <- matrix(0, nrow=f, ncol=k_len)
SEG1 <- matrix(0, nrow=f, ncol=k0)
SEG2 <- matrix(0, nrow=f, ncol=k1)
storage.mode(S) <- "integer"
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##データの設定
index_allocation <- matrix(1:(k_len*k1), ncol=k_len)
vec1_list <- list()
vec2_list <- list()
for(i in 1:d){
  vec1_list[[i]] <- rep(1, w[i])
}
for(j in 1:v){
  vec2_list[[j]] <- rep(1,  sum(WX[, j]))
}
wsum01 <- matrix(0, nrow=d, ncol=k0)
wsum02 <- matrix(0, nrow=d, ncol=k1)
vf01 <- matrix(0, nrow=k0, k_len)
vf02 <- matrix(0, nrow=k1*k_len, v)


##対数尤度の基準値
#ユニグラムモデルの対数尤度
par0 <- colSums(WX) / f + beta02
par <- par0 / sum(par0)
LLst <- sum(WX %*% log(par))


####マルコフ連鎖モンテカルロ法でパラメータをサンプリング####
for(rp in 1:R){
  
  ##下位階層の潜在変数の割当をサンプリング
  #上位トピックの尤度を推定
  LLi <- matrix(0, nrow=f, ncol=k_len)
  word_par <- list()
  for(j in 1:k_len){
    word_par[[j]] <- burden_fr(theta2, phi2[[j]], wd, w, k1)
    LLi[, j] <- rowSums(word_par[[j]]$Bur)
  }

  #潜在変数の割当をサンプリング
  allocation_par <- r * LLi
  s_rate <- allocation_par / rowSums(allocation_par)   #潜在変数の割当確率
  Si <- rmnom(f, 1, s_rate)   #多項分布から潜在変数をサンプリング
  s_vec <- as.numeric(Si %*% 1:k_len)
  
  
  ##上位トピックをサンプリング
  #上位トピックの尤度と割当確率を推定
  word_par1 <- burden_fr(theta1, phi1, s_vec, w, k0)
  z1_rate <- word_par1$Br
  
  #多項分布から上位トピックをサンプリング
  Zi1 <- rmnom(f, 1, z1_rate)
  z1_vec <- as.numeric(Zi1 %*% 1:k0)
  r <- phi1[z1_vec, ]   #混合率の更新

  
  ##下位トピックをサンプリング
  Zi2 <- matrix(0, nrow=f, ncol=k1)
  index_s <- list()
  for(j in 1:k_len){
    index <- which(Si[, j]==1)
    index_s[[j]] <- index
    z2_rate <- word_par[[j]]$Br[index, ] 
    Zi2[index, ] <- rmnom(length(index), 1, z2_rate)   
  }


  ##トピック分布をサンプリング
  #ディリクレ分布のパラメータを推定
  for(i in 1:d){
    wsum01[i, ] <- vec1_list[[i]] %*% Zi1[doc_list[[i]], ] + alpha01
    wsum02[i, ] <- vec1_list[[i]] %*% Zi2[doc_list[[i]], ] + alpha02
  }
  #ディリクレ分布からトピック分布をサンプリング
  theta1 <- extraDistr::rdirichlet(d, wsum01)
  theta2 <- extraDistr::rdirichlet(d, wsum02)

  
  ##単語分布をサンプリング
  #ディリクレ分布のパラメータを推定
  n <- colSums(Si)
  for(j in 1:k_len){vf01[, j] <- rep(1, n[j]) %*% Zi1[index_s[[j]], ]}
  for(j in 1:v){vf02[, j] <- as.numeric(t(Zi2[wd_list[[j]], , drop=FALSE]) %*% Si[wd_list[[j]], ])}
  vf1 <- vf01 + beta01
  vf2 <- vf02 + beta02
  
  
  #ディリクレ分布から単語分布をサンプリング
  phi1 <- extraDistr::rdirichlet(k0, vf1)
  phi0 <- extraDistr::rdirichlet(k_len*k1, vf2)
  phi2 <- list()
  for(j in 1:k_len){phi2[[j]] <- phi0[index_allocation[, j], ]}

  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi0
     
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      S <- S + Si
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      LL <- sum(log(rowSums(LLi * Si)))
      print(c(LL, LLst))
      print(round(c(exp(-LL / f), exp(-LLst / f)), 3))
      print(round(cbind(phi1, phit01), 3))
      print(round(cbind(theta2[1:10, ], thetat02[1:10, ]), 3))
      print(rbind(si_freq=colSums(Si), s_freq))
    }
  }
}

####サンプリング結果の可視化と要約####
burnin <- 2000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#上位階層のトピック分布の可視化
matplot(t(THETA1[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[2, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#上位階層の割当確率の分布の可視化
matplot(t(PHI1[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI1[2, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI1[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#下位階層のトピック分布の可視化
matplot(t(THETA2[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[2, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[4, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#下位階層の単語分布の可視化
matplot(t(PHI2[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI2[10, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI2[20, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI2[30, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI2[40, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI2[50, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")


##サンプリング結果の要約推定量
#上位トピック分布の事後推定量
topic_mu1 <- apply(THETA1[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu1, thetat01), 3)
round(topic_sd1 <- apply(THETA1[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#上位階層の割当確率の事後推定量
word_mu1 <- apply(PHI1[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(t(rbind(word_mu1, phit01)), 3)

#下位トピック分布の事後推定量
topic_mu2 <- apply(THETA2[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu2, thetat02), 3)
round(topic_sd2 <- apply(THETA2[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#下位階層の単語出現確率の事後推定量
word_mu2 <- apply(PHI2[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(t(word_mu2), 3)


##割当られたトピックの事後分布
round(cbind(SEG1 / rowSums(SEG1), Z1), 3)
round(cbind(SEG2 / rowSums(SEG2), apply(SEG1 / rowSums(SEG1), 1, which.max), Z1, Z2), 3)


