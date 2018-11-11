#####Nested Chinese Restaurant Process LDA#####
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
L <- 3   #階層数
k1 <- kt1 <- 1   #レベル1の階層数
k2 <- kt2 <- 4   #レベル2の階層数
k3 <- kt3 <- rtpois(k2, 3, a=1, b=Inf)   #レベル3の階層数
k <- sum(c(k1, k2, k3))   #総トピック数
d <- 4000   #文書数
v <- 1000   #語彙数
index_v1 <- 1:200; v1 <- length(index_v1)
index_v2 <- 201:550; v2 <- length(index_v2)
index_v3 <- 551:v; v3 <- length(index_v3)
w <- rpois(d, rgamma(d, 50, 0.5))   #単語数
f <- sum(w)   #単語数

#IDを設定
d_id <- rep(1:d, w)

##データの生成
for(rp in 1:1000){
  print(rp)
  
  #ディレクリ分布のパラメータを設定
  alpha1 <- alphat1 <- c(0.1, 0.2, 0.25)
  alpha2 <- alphat2 <- rep(10.0, k2)
  alpha3 <- alphat3 <- list()
  for(j in 1:k2){
    alpha3[[j]] <- alphat3[[j]] <- rep(3.0, k3[j])
  }
  beta1 <- betat1 <- c(rep(3.0, v1), rep(0.005, v2), rep(0.005, v3))
  beta2 <- betat2 <- c(rep(0.005, v1), rep(0.3, v2), rep(0.005, v3))
  beta3 <- betat3 <- c(rep(0.005, v1), rep(0.005, v2), rep(0.1, v3))
  
  #ディレクリ分布からパラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(d, alpha1)
  theta2 <- thetat2 <- as.numeric(extraDistr::rdirichlet(1, alpha2))
  theta3 <- thetat3 <- list()
  for(j in 1:k2){
    theta3[[j]] <- thetat3[[j]] <- as.numeric(extraDistr::rdirichlet(1, alpha3[[j]]))
  }
  phi1 <- phit1 <- as.numeric(extraDistr::rdirichlet(k1, beta1))
  phi2 <- phit2 <- extraDistr::rdirichlet(k2, beta2)
  phi3 <- phit3 <- list()
  for(j in 1:k2){
    phi3[[j]] <- phit3[[j]] <- extraDistr::rdirichlet(k3[j], beta3)
  }
  phi <- phit <- rbind(phi1, phi2, do.call(rbind, phi3))
  
  ##生成過程に基づき単語を生成
  Z1 <- matrix(0, nrow=d, ncol=L)
  Z1[, 1] <- 1
  Z12_list <- list()
  Z13_list <- list()
  Z2_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  data_list <- list()
  word_list <- list()
  
  for(i in 1:d){
    #ノードを生成
    z12 <- rmnom(1, 1, theta2) 
    Z1[i, 2] <- as.numeric(z12 %*% 1:k2)
    z13 <- rmnom(1, 1, theta3[[Z1[i, 2]]])
    Z1[i, 3] <- as.numeric(z13 %*% 1:k3[Z1[i, 2]])
    
    #トピックのレベルを生成
    z2 <- rmnom(w[i], 1, theta1[i, ])
    z2_vec <- as.numeric(z2 %*% 1:L)
    
    #レベルごとに単語を生成
    index <- list()
    words <- matrix(0, nrow=w[i], ncol=v)
    for(j in 1:L){
      index[[j]] <- which(z2_vec==j)
    }
    words[index[[1]], ] <- rmnom(length(index[[1]]), 1, phi1)
    words[index[[2]], ] <- rmnom(length(index[[2]]), 1, phi2[Z1[i, 2], ])
    words[index[[3]], ] <- rmnom(length(index[[3]]), 1, phi3[[Z1[i, 2]]][Z1[i, 3], ])  
    
    
    #データを格納
    Z12_list[[i]] <- z12
    Z2_list[[i]] <- z2
    WX[i, ] <- colSums(words)
    data_list[[i]] <- words
    word_list[[i]] <- as.numeric(words %*% 1:v)
  }
  if(min(colSums(WX)) > 0) break
}

#リストを変換
Z12 <- do.call(rbind, Z12_list)
Z2 <- do.call(rbind, Z2_list)
word_vec <- unlist(word_list)
Data <- do.call(rbind, data_list)
storage.mode(Data) <- "integer"
storage.mode(Z2) <- "integer"
storage.mode(WX) <- "integer"
sparse_wx <- as(WX, "CsparseMatrix")
sparse_data <- as(Data, "CsparseMatrix")
rm(data_list); rm(Z2_list); rm(word_list)
gc(); gc()


####マルコフ連鎖モンテカルロ法でnCRP-LDAを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / rowSums(Bur)   #負担率
  r <- colSums(Br) / sum(Br)   #混合率
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##アルゴリズムの設定
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10
g <- 25
sensor <- 1

##事前分布の設定
alpha1 <- 1
alpha2 <- 1
alpha3 <- 1
beta1 <- 1   #CRPの事前分布
beta2 <- 1

##パラメータの真値
#トピック数の真値
k1 <- kt1; k2 <- kt2; k3 <- kt3
index_c <- rep(1:k2, k3)

#トピックモデルのパラメータの真値
theta <- thetat1
phi1 <- (phit1 + 10^-100) / sum(phit1 + 10^-100)
phi2 <- (phit2 + 10^-100) / rowSums(phit2 + 10^-100)
phi3 <- (do.call(rbind, phit3) + 10^-100) / rowSums(do.call(rbind, phit3) + 10^-100) 
Zi12 <- matrix(as.numeric(table(1:d, Z1[, 2])), nrow=d, ncol=k2)
Zi1 <- matrix(0, nrow=d, ncol=sum(k3))
for(j in 1:k2){
  Zi1[, index_c==j] <- Zi12[, j] * matrix(as.numeric(table(1:d, Z1[, 3])), nrow=d, ncol=max(k3))[, 1:k3[j]]
}
ZT2 <- Zi2 <- Z2
ZT1 <- Zi1

#ベストの対数尤度
LLi <- cbind(phit1[word_vec], rowSums(t(phit2)[word_vec, ] * Zi12[d_id, ]), 
             rowSums(t(do.call(rbind, phit3))[word_vec, ] * Zi1[d_id, ]))
LLbest <- sum(log(rowSums(LLi * Zi2)))


#トピック数の初期値
k1 <- 1; k2 <- 3; k3 <- rep(1, k2)

#パラメータの初期値
theta <- extraDistr::rdirichlet(d, rep(1, L))
phi1 <- as.numeric(extraDistr::rdirichlet(1, rep(100.0, v)))
phi2 <- extraDistr::rdirichlet(k2, rep(100.0, v))
phi3 <- extraDistr::rdirichlet(sum(k3), rep(100.0, v))

#ノード割当とレベル割当の初期値
Zi1 <- rmnom(d, 1, rep(1/sum(k3), sum(k3)))
Zi2 <- rmnom(f, 1, rep(1/L, L))

##パラメータの格納用配列
max_seg1 <- 15
max_seg2 <- 50
PHI1 <- matrix(0, nrow=R/keep, ncol=v)
PHI2 <- array(0, dim=c(max_seg1, v, R/keep))
PHI3 <- array(0, dim=c(max_seg2, v, R/keep))
THETA <- array(0, dim=c(d, L, R/keep))
SEG1 <- matrix(0, nrow=d, ncol=max_seg2)
SEG2 <- matrix(0, nrow=f, ncol=L)

#基底分布の設定
G0 <- rep(0.5, v)
G0 <- colSums(sparse_data)/v + 1


##インデックスを作成
doc_vec <- doc_list <- list()
for(i in 1:d){
  doc_list[[i]] <- which(d_id==i)
  doc_vec[[i]] <- rep(1, length(doc_list[[i]]))
}


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##レベル1のパスの対数尤度
  if(rp > burnin){
    g <- 3
  }
  LLho1 <-  (sparse_data * Zi2[, 1]) %*% log(phi1)
  
  ##レベル2のパスの対数尤度
  #新しい潜在変数のパラメータを結合
  index_d <- sample(1:d, 1)
  par01 <- extraDistr::rdirichlet(g, G0)
  phi2_new <- rbind(phi2, par01)
  
  #文書ごとに対数尤度を設定
  LLho2 <- (sparse_data * Zi2[, 2]) %*% t(log(phi2_new))
  LLho2 <- LLho2[, c(1:k2, which.max(colSums(LLho2[, -(1:k2)]))+length(1:k2))]
  
  ##レベル3のパスの対数尤度
  #パスのルートインデックスを作成
  index_c <- rep(1:(k2+1), c(k3+1, 1))
  new_c <- c()
  for(j in 1:(k2+1)){
    new_c <- c(new_c, max(which(index_c==j)))
  }
  #新しい潜在変数のパラメータを結合
  phi3_new <- matrix(0, nrow=v, length(index_c))
  phi3_new[, -new_c] <- t(phi3)
  
  #文書ごとに対数尤度を設定
  LLho3 <- matrix(0, nrow=f, ncol=length(index_c))
  for(j in 1:(k2+1)){
    if(j < (k2+1)){
      index <- index_c==j
      phi_new <- cbind(phi3_new[, index][, -sum(index)], t(extraDistr::rdirichlet(g, G0)))
      LLho0 <- (sparse_data * Zi2[, 3]) %*% log(phi_new)
      LLho3[, index] <- as.matrix(LLho0[, c(1:k3[j], which.max(colSums(LLho0[, -(1:k3[j])]))+k3[j])])
  
    } else {
  
      index <- index_c==j
      phi_new <- t(extraDistr::rdirichlet(g, G0))
      LLho0 <- (sparse_data * Zi2[, 3]) %*% log(phi_new)
      LLho3[, index] <- LLho0[, which.max(colSums(LLho0))]
    }
  }
  
  ##nCPRからパス変数をサンプリング
  #文書ごとにレベルごとの対数尤度の和をとる
  LLho <- as.matrix(t(LLho1 + LLho2[, index_c] + LLho3))
  LLi0 <- matrix(0, nrow=d, ncol=length(index_c))
  for(i in 1:d){
    LLi0[i, ] <- LLho[, doc_list[[i]]] %*% doc_vec[[i]]
  }
  LLi <- exp(LLi0 - rowMaxs(LLi0))   #尤度に変換

  #nCRPのパラメータを設定
  N <- matrix(beta1, nrow=d, ncol=length(index_c))
  N[, -new_c] <- matrix(colSums(Zi1), nrow=d, ncol=ncol(Zi1), byrow=T) - Zi1
  gamma0 <- N / (d + beta1 - 1)   #nCPRの事前分布

  #パス変数の割当確率からパスをサンプリング
  gamma <- gamma0 * LLi   #パス割当の潜在確率
  Zi0 <- rmnom(d, 1, gamma / rowSums(gamma))   #多項分布からパスをサンプリング
  
  #生成が一定以下のパスを削除
  if(rp > burnin){
    sensor <- 25
  } else {
    sensor <- 1
  }
  index_z1 <- colSums(Zi0) >= sensor
  Zi1 <- Zi0[, index_z1]; Zi1_T <- t(Zi1); index_c <- index_c[index_z1]
  k2 <- length(unique(index_c)); k3 <- as.numeric(table(index_c))

  ##単語分布のパラメータをサンプリング
  if(rp > burnin){
    alpha3 <- 0.1
  }
  #レベル1の単語分布をサンプリング
  vsum11 <- colSums(sparse_data * Zi2[, 1]) + alpha3
  phi1 <- as.numeric(extraDistr::rdirichlet(1, vsum11))
  
  #レベル2の単語分布をサンプリング
  phi2 <- matrix(0, nrow=k2, ncol=v)
  Zi12 <- matrix(0, nrow=d, ncol=k2)
  for(j in 1:k2){
    Zi12[, j] <- rowSums(Zi1[, index_c==j, drop=FALSE])
    vsum12 <- colSums((Zi12[d_id, j] * Zi2[, 2]) * sparse_data) + alpha3
    phi2[j, ] <- extraDistr::rdirichlet(1, vsum12)
  }

  #レベル3の単語分布をサンプリング
  vsum13 <- as.matrix(t(Zi1[d_id, ]) %*% (sparse_data * Zi2[, 3]) + alpha3)
  phi3 <- extraDistr::rdirichlet(nrow(vsum13), vsum13)
  

  ##トピック割当をサンプリング
  #トピック割当の尤度を設定
  par1 <- phi1[word_vec]   
  par2 <- rowSums(t(phi2)[word_vec, , drop=FALSE] * Zi12[d_id, ])
  par3 <- rowSums(t(phi3)[word_vec, , drop=FALSE] * Zi1[d_id, ])
  z_par <- theta[d_id, ] * cbind(par1, par2, par3)   #レベルごとのトピック尤度
  
  #多項分布からトピック割当をサンプリング
  z_rate <- z_par / rowSums(z_par)   #レベル割当確率
  Zi2 <- rmnom(f, 1, z_rate)   #多項分布からレベル割当を生成
  Zi2_T <- t(Zi2)
  
  ##トピック分布を更新
  #ディレクリ分布からトピック分布をサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=L)
  for(i in 1:d){
    wsum0[i, ] <- Zi2_T[, doc_list[[i]]] %*% doc_vec[[i]]
  }
  wsum <- wsum0 + alpha2   #ディリクレ分布のパラメータ
  theta <- extraDistr::rdirichlet(d, wsum)   #パラメータをサンプリング
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    mkeep <- 1
    PHI1[mkeep, ] <- phi1
    PHI2[1:nrow(phi2), , mkeep] <- phi2
    PHI3[1:nrow(phi3), , mkeep] <- phi3
    THETA[, , mkeep] <- theta
  } 

  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG1[, 1:ncol(Zi1)] <- SEG1[, 1:ncol(Zi1)] + Zi1
    SEG2 <- SEG2 + Zi2
  }
    
  if(rp%%disp==0){
    #サンプリング結果を確認
    print(rp)
    print(c(k2, k3))
    print(c(kt2, kt3))
    print(colSums(Zi1))
    print(c(sum(log(rowSums(z_par))), LLbest))
  }
}

sum(as.numeric(Zi2 %*% 1:3) == as.numeric(Z2 %*% 1:3))
colSums(Zi2)
colSums(Z2)
