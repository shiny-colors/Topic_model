#####Nested Latent Dirichlet Allocation#####
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
##データの設定
L <- 3   #階層数
k1 <- 1   #レベル1の階層数
k2 <- 4   #レベル2の階層数
k3 <- rtpois(k2, 3, a=1, b=Inf)   #レベル3の階層数
k <- sum(c(k1, k2, k3))   #総トピック数
d <- 2000   #文書数
v1 <- 300
v2 <- 300
v3 <- 400
v <- v1 + v2 + v3   #語彙数
w <- rpois(d, rgamma(d, 85, 0.5))   #単語数
f <- sum(w)   #単語数


#IDを設定
d_id <- rep(1:d, w)

##データの生成
for(rp in 1:1000){
  print(rp)
  
  #ディレクリ分布のパラメータを設定
  alpha1 <- alphat1 <- c(0.2, 0.25, 0.3)
  alpha2 <- alphat2 <- rep(10.0, k2)
  alpha3 <- alphat3 <- list()
  for(j in 1:k2){
    alpha3[[j]] <- alphat3[[j]] <- rep(3.0, k3[j])
  }
  beta1 <- c(rep(1.0, v1), rep(0.001, v2+v3))
  beta2 <- c(rep(0.0001, v1), rep(0.2, v2), rep(0.0001, v3))
  beta3 <- c(rep(0.0001, v1+v2), rep(0.15, v3))

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


##事前分布の設定
alpha1 <- 1
alpha2 <- 1
alpha3 <- 0.1
beta1 <- 1   #CRPの事前分布
beta2 <- 1

##パラメータの真値
#トピックモデルのパラメータの真値
theta <- thetat1
phi1 <- phit1; phi2 <- phit2; phi3 <- phit3
phi2[phi2==0] <- 10^-100
for(j in 1:k2){
  phi3[[j]][phi3[[j]]==0] <- 10^-10
}
Zi2 <- Z2
Zi1 <- rmnom(d, 1, rep(1/sum(k3), sum(k3)))
Zi12 <- matrix(0, nrow=d, ncol=k2)
cumsum_k3 <- cumsum(k3)
index_z1 <- list()
for(j in 1:length(k3)){
  if(j==1){
    index_z1[[j]] <- 1:cumsum_k3[j]
  } else {
    index_z1[[j]] <- (cumsum_k3[j-1]+1):cumsum_k3[j]
  }
  Zi12[, j] <- rowSums(Zi1[, index_z1[[j]]])
}
r <- colMeans(Zi1)


#パラメータの初期値
theta <- extraDistr::rdirichlet(d, rep(10, L))
phi1 <- extraDistr::rdirichlet(k1, rep(100, v))
phi2 <- extraDistr::rdirichlet(k2, rep(100, v))
phi3 <- list()
for(j in 1:k2){
  phi3[[j]] <- extraDistr::rdirichlet(k3[j], rep(100.0, v))
}

#トピック割当の初期値
Zi1 <- rmnom(d, 1, rep(1/sum(k3), sum(k3)))
Zi12 <- matrix(0, nrow=d, ncol=k2)
cumsum_k3 <- cumsum(k3)
index_z1 <- list()
for(j in 1:length(k3)){
  if(j==1){
    index_z1[[j]] <- 1:cumsum_k3[j]
  } else {
    index_z1[[j]] <- (cumsum_k3[j-1]+1):cumsum_k3[j]
  }
  Zi12[, j] <- rowSums(Zi1[, index_z1[[j]], drop=FALSE])
}
Zi2 <- rmnom(f, 1, rep(1/L, L))
r <- colMeans(Zi1)

##パラメータの格納用配列
THETA <- array(0, dim=c(d, L, R/keep))
PHI1 <- matrix(0, nrow=R/keep, ncol=v)
PHI2 <- array(0, dim=c(k2, v, R/keep))
PHI3 <- array(0, dim=c(ncol(Zi1), v, R/keep))
SEG1 <- matrix(0, nrow=d, ncol=ncol(Zi1))
SEG2 <- matrix(0, nrow=f, ncol=L)

##インデックスを作成
doc_vec <- doc_list <- list()
for(i in 1:d){
  doc_list[[i]] <- which(d_id==i)
  doc_vec[[i]] <- rep(1, length(doc_list[[i]]))
}


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##レベル2のパスの対数尤度
  #文書ごとに対数尤度を設定
  LLho2 <- as.matrix(t((sparse_data * Zi2[, 2]) %*% t(log(phi2))))
  LLi2_T <- matrix(0, ncol=d, nrow=nrow(phi2))
  for(i in 1:d){
    LLi2_T[, i] <- LLho2[, doc_list[[i]]] %*% doc_vec[[i]]
  }
  LLi2 <- t(LLi2_T)
  Li2 <- exp(LLi2 - rowMaxs(LLi2))   #尤度に変換

  
  ##レベル3のパスの対数尤度
  LLi3 <- list()
  LLi_z <- cbind()
  
  #レベル2のパスごとにレベル3のパスの対数尤度を設定
  for(j in 1:k2){
  
    #文書ごとに対数尤度を設定
    LLho3 <- as.matrix(t((sparse_data * Zi2[, 3]) %*% t(log(phi3[[j]]))))
    LLi3_T <- matrix(0, nrow=k3[j], ncol=d)
    for(i in 1:d){
      LLi3_T[, i] <- LLho3[, doc_list[[i]]] %*% doc_vec[[i]]
    }
    LLi3[[j]] <- t(LLi3_T)
    LLi_z <- cbind(LLi_z, LLi3[[j]] + LLi2[, j])   #対数尤度の和
  }
  Li_z <- exp(LLi_z - rowMaxs(LLi_z))   #尤度に変換

  
  #潜在変数zの割当確率の推定とZのサンプリング
  gamma <- matrix(r, nrow=d, ncol=ncol(Zi1), byrow=T) * Li_z
  z1_rate <- gamma / rowSums(gamma)
  Zi1 <- rmnom(d, 1, z1_rate)
  
  #レベル2のトピック割当を設定
  Zi12 <- matrix(0, nrow=d, ncol=k2)
  for(j in 1:k2){
    k3[j] <- length(index_z1[[j]])
    Zi12[, j] <- rowSums(Zi1[, index_z1[[j]], drop=FALSE])
  }
  
  #混合率の更新
  r <- colMeans(Zi1)

  
  ##パスごとに単語分布のパラメータを更新
  #レベル1の単語分布をサンプリング
  vsum11 <- colSums(sparse_data * Zi2[, 1]) + alpha3
  phi1 <- as.numeric(extraDistr::rdirichlet(1, vsum11))
  
  #レベル2の単語分布をサンプリング
  vsum12 <- as.matrix(t(Zi12[d_id, ]) %*% (sparse_data * Zi2[, 2])) + alpha3
  phi2 <- extraDistr::rdirichlet(nrow(vsum12), vsum12)
  
  #レベル3の単語分布をサンプリング
  phi3 <- list()
  for(j in 1:k2){
    vsum13 <- as.matrix(t(Zi1[d_id, index_z1[[j]], drop=FALSE]) %*% (sparse_data * Zi2[, 3]) + alpha3)
    phi3[[j]] <- extraDistr::rdirichlet(nrow(vsum13), vsum13)
  }
  
  
  ##トピック割当をサンプリング
  #トピック割当の尤度を設定
  par1 <- phi1[word_vec]   
  par2 <- rowSums(t(phi2)[word_vec, , drop=FALSE] * Zi12[d_id, ])
  par3 <- matrix(0, nrow=f, ncol=k2)
  for(j in 1:k2){
    par3[, j] <- rowSums(t(phi3[[j]])[word_vec, , drop=FALSE] * Zi1[d_id, index_z1[[j]], drop=FALSE])
  }
  z_par <- theta[d_id, ] * cbind(par1, par2, rowSums(par3))   #レベルごとのトピック尤度
  
  #多項分布からトピック割当をサンプリング
  z2_rate <- z_par / rowSums(z_par)   #レベル割当確率
  Zi2 <- rmnom(f, 1, z2_rate)   #多項分布からレベル割当を生成
  Zi2_T <- t(Zi2)
  
  
  ##トピック分布を更新
  #ディレクリ分布のパラメータ
  wsum0 <- matrix(0, nrow=d, ncol=L)
  for(i in 1:d){
    wsum0[i, ] <- Zi2_T[, doc_list[[i]]] %*% doc_vec[[i]]
  }
  wsum <- wsum0 + alpha2 
  
  #ディレクリ分布からパラメータをサンプリング
  theta <- extraDistr::rdirichlet(d, wsum)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI1[mkeep, ] <- phi1
    PHI2[, , mkeep] <- phi2
    PHI3[, , mkeep] <- do.call(rbind, phi3)
     
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    if(rp%%disp==0){
      #サンプリング結果を確認
      print(rp)
      print(sum(log(rowSums(z_par))))
      print(colSums(Zi1))
      print(round(cbind(theta[1:10, ], thetat1[1:10, ]), 3))
      print(round(rbind(phi2[1:nrow(phi2), 1:20], phit2[, 1:20]), 3))
      print(round(rbind(phi1[1:40], phit1[1:40]), 3))
    }
  }
}

matplot(PHI1, type="l", xlab="サンプリング回数", ylab="パラメータ", main="レベル1の単語分布のサンプリング結果")
matplot(t(PHI2[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="レベル2の単語分布のサンプリング結果")
matplot(t(PHI2[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="レベル2の単語分布のサンプリング結果")
matplot(t(PHI2[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="レベル2の単語分布のサンプリング結果")
matplot(t(PHI2[4, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="レベル2の単語分布のサンプリング結果")
matplot(t(PHI3[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="レベル2の単語分布のサンプリング結果")
