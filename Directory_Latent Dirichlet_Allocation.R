#####ディレクトリLDAモデル#####
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
#set.seed(2506787)

####データの発生####
##データの設定
k <- 10   #トピック数
dir1 <- 5   #上位ディレクトリ数
dir2 <- 20   #下位ディレクトリ数
d <- 15000   #文書数
v11 <- 150   #ディレクトリあたりの語彙数
v12 <- dir1*v11   #ディレクトリ構造に関係のある語彙数
v2 <- 500   #ディレクトリ構造に関係のない語彙数
v <- v12 + v2   #総語彙数
index_v11 <- matrix(1:v12, nrow=dir1, ncol=v11, byrow=T)
dir_v <- matrix(1:v12, nrow=dir1, ncol=v11, byrow=T)   #ディレクトリの単語構造
w <- rpois(d, rgamma(d, 65, 0.5))   #文書あたりの単語数
f <- sum(w)   #総単語数

##IDの設定
d_id <- rep(1:d, w)
t_id_list <- list()
for(i in 1:d){
  t_id_list[[i]] <- 1:w[i]
}
t_id <- unlist(t_id_list)

##ディレクトリの割当を設定
#上位と下位のディレクトリ割当を設定
dir_sets1 <- matrix(1:dir2, nrow=dir1, ncol=dir2/dir1, byrow=T)
dir_sets2 <- rep(1:dir1, rep(dir2/dir1, dir1))
dir_freq <- rtpois(d, 0.7, 0, 3)   #文書あたりのディレクトリ数
dir_id <- rep(1:d, dir_freq)   #ディレクトリのid
dir_n <- length(dir_id)
dir_index <- list()
for(i in 1:d){
  dir_index[[i]] <- which(dir_id==i)
}

#ディレクトリの生成
dir_data1 <- matrix(0, nrow=dir_n, ncol=dir1)
dir_data2 <- matrix(0, nrow=dir_n, ncol=dir2)
for(i in 1:d){
  repeat{
    x <- rmnom(dir_freq[i], 1, rep(1, dir2))
    if(max(colSums(x))==1){
      index <- which(dir_id==i)
      x <- x[order(as.numeric(x %*% 1:dir2)), , drop=FALSE]
      dir_data1[index, ] <- t(apply(x, 1, function(y) tapply(y, dir_sets2, sum)))
      dir_data2[index, ] <- x
      break
    }
  }
}
#ディレクトリをベクトルに変換
dir_vec1 <- as.numeric(dir_data1 %*% 1:dir1)
dir_vec2 <- as.numeric(dir_data2 %*% 1:dir2)


##パラメータの設定
#ディリクレ分布の事前分布
alpha11 <- rep(0.15, k)
alpha21 <- c(rep(0.025, length(1:v12)), rep(0.15, length(1:v2)))
alpha22 <- matrix(0, nrow=dir2, ncol=v)
for(j in 1:dir1){
  x <- rep(0.001, v)
  x[index_v11[j, ]] <- 0.2
  alpha22[dir_sets1[j, ], ] <- matrix(x, nrow=length(dir_sets1[j, ]), ncol=v, byrow=T)
}

##すべての単語が出現するまでデータの生成を続ける
for(rp in 1:1000){
  print(rp)
  
  #ディリクレ分布からパラメータを生成
  theta <- thetat <- extraDistr::rdirichlet(d, alpha11)
  phi1 <- phit1 <- extraDistr::rdirichlet(k, alpha21)
  phi2 <- phit2 <- extraDistr::rdirichlet(dir2, alpha22)
  
  #スイッチング変数を生成
  gamma_list <- list()
  for(i in 1:d){
    if(dir_freq[i]==1){
      par <- c(6.25, 5.0)
      gamma_list[[i]] <- rbeta(1, par[1], par[2])
    } else {
      par <- c(5.0, runif(dir_freq[i], 1.0, 4.5))
      gamma_list[[i]] <- as.numeric(extraDistr::rdirichlet(1, par))
    }
  }
  
  ##モデルに基づきデータを生成
  word_list <- wd_list <- Z1 <- z1_list <- z2_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    #スイッチング変数を生成
    n <- dir_freq[i] + 1
    if(dir_freq[i]==1){
      z1 <- rbinom(w[i], 1, gamma_list[[i]])
      Z1[[i]] <- cbind(z1, 1-z1)
      z1_list[[i]] <- as.numeric((Z1[[i]] * matrix(c(1, dir_vec2[dir_index[[i]]]), nrow=w[i], ncol=n, byrow=T)) %*% rep(1, n))
    } else {
      Z1[[i]] <- rmnom(w[i], 1, gamma_list[[i]])
      z1_list[[i]] <- as.numeric((Z1[[i]] * matrix(c(1, dir_vec2[dir_index[[i]]]), nrow=w[i], ncol=n, byrow=T)) %*% rep(1, n))
    }
    
    #多項分布よりトピックを生成
    z2 <- matrix(0, nrow=w[i], ncol=k)
    index <- which(Z1[[i]][, 1]==1)
    z2[index, ] <- rmnom(length(index), 1, theta[i, ])
    z2_vec <- as.numeric(z2 %*% 1:k)
    
    #トピックおよびディレクトリから単語を生成
    word <- matrix(0, nrow=w[i], ncol=v)
    word[index, ] <- rmnom(length(index), 1, phi1[z2_vec[index], ])   #トピックから単語を生成
    word[-index, ] <- rmnom(w[i]-length(index), 1, phi2[z1_list[[i]][-index], ])   #ディレクトリから単語を生成
    wd <- as.numeric(word %*% 1:v)
    storage.mode(word) <- "integer"
    
    #データを格納
    z2_list[[i]] <- z2
    wd_list[[i]] <- wd
    word_list[[i]] <- word
    WX[i, ] <- colSums(word)
  }
  #全単語が出現していたらbreak
  if(min(colSums(WX) > 0)) break
}

##リストを変換
wd <- unlist(wd_list)
Z2 <- do.call(rbind, z2_list)
z2_vec <- as.numeric(Z2 %*% 1:k)
sparse_data <- sparseMatrix(1:f, wd, dims=c(f, v))
sparse_data_T <- t(sparse_data)
rm(word_list); rm(wd_list); rm(z2_list)
gc(); gc()

#スイッチング変数の真値
ZT1_list <- list()
for(i in 1:d){
  ZT1_list[[i]] <- Z1[[i]][, 1]
}
ZT1 <- unlist(ZT1_list)

#####マルコフ連鎖モンテカルロ法でDLDAを推定####
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
alpha1 <- 0.1
alpha2 <- 0.1
beta1 <- 1
beta2 <- 1

##真値の設定
theta <- thetat
phi1 <- phit1
phi2 <- phit2
gamma <- gamma_list

##初期値の設定
#トピック分布の初期値
theta <- extraDistr::rdirichlet(d, rep(1.0, k))
phi1 <- extraDistr::rdirichlet(k, rep(1.0, v))
phi2 <- extraDistr::rdirichlet(dir2, rep(1.0, v))

#スイッチング分布の初期値
gamma <- list()
for(i in 1:d){
  if(dir_freq[i]==1){
    gamma[[i]] <- 0.5
  } else {
    n <- dir_freq[i]+1
    gamma[[i]] <- rep(1/n, n)
  }
}

##パラメータの保存用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI1 <- array(0, dim=c(k, v, R/keep))
PHI2 <- array(0, dim=c(dir2, v, R/keep))
SEG11 <- rep(0, f)
SEG12 <- matrix(0, nrow=f, ncol=dir2)
SEG2 <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG12) <- "integer"
storage.mode(SEG2) <- "integer"

##インデックスを設定
doc_list <- doc_vec <- list()
wd_list <- wd_vec <- list()
for(i in 1:d){
  doc_list[[i]] <- which(d_id==i)
  doc_vec[[i]] <- rep(1, length(doc_list[[i]]))
}
for(j in 1:v){
  wd_list[[j]] <- which(wd==j)
  wd_vec[[j]] <- rep(1, length(wd_list[[j]]))
}

##データの設定
dir_z <- matrix(0, nrow=d, ncol=dir2)
dir_list2 <- dir_list1 <- list()
for(i in 1:d){
  dir_z[i, ] <- colSums(dir_data2[dir_index[[i]], , drop=FALSE])
  dir_list1[[i]] <- (dir_z[i, ] * 1:dir2)[dir_z[i, ] > 0]
  dir_list2[[i]] <- matrix(dir_list1[[i]], nrow=w[i], ncol=dir_freq[i], byrow=T)
}
dir_Z <- dir_z[d_id, ]
storage.mode(dir_Z) <- "integer"

##対数尤度の基準値
LLst <- sum(sparse_data %*% log(colMeans(sparse_data)))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとのスイッチング変数を生成
  #トピックとディレクトリの期待尤度
  Lho1 <- theta[d_id, ] * t(phi1)[wd, ]
  Li1 <- rowSums(Lho1)   #トピックの期待尤度
  Li2 <- t(phi2)[wd, ] * dir_Z   #ディレクトリの尤度
  
  #ベルヌーイ分布あるいは多項分布よりスイッチング変数を生成
  Zi11 <- Zi12 <- rep(0, f)
  Lho_list <- list()
  
  for(i in 1:d){
    if(dir_freq[i]==1){
  
      #潜在変数zの設定
      omega <- matrix(c(gamma[[i]], 1-gamma[[i]]), nrow=w[i], ncol=dir_freq[i]+1, byrow=T)
      z_par <- omega * cbind(Li1[doc_list[[i]]], Li2[doc_list[[i]], dir_list1[[i]]])
      z_rate <- z_par[, 1] / rowSums(z_par)
      Lho_list[[i]] <- z_par
      
      
      #ベルヌーイ分布よりスイッチング変数を生成
      z1 <- rbinom(w[i], 1, z_rate)
      Zi11[doc_list[[i]]] <- z1   #トピックに関係のある単語
      Zi12[doc_list[[i]]] <- (1-z1) * dir_list1[[i]]   #ディレクトリに関係のある単語
      
      #ベータ分布から混合率をサンプリング
      z_freq <- t(z1) %*% doc_vec[[i]]
      gamma[[i]] <- rbeta(1, z_freq+beta1, w[i]-z_freq+beta2)
  
    } else {
      
      #潜在変数zの設定
      omega <- matrix(gamma[[i]], nrow=w[i], ncol=dir_freq[i]+1, byrow=T)
      z_par <- omega * cbind(Li1[doc_list[[i]]], Li2[doc_list[[i]], dir_list1[[i]]])
      z_rate <- z_par / rowSums(z_par)
      Lho_list[[i]] <- z_par
      
      z1 <- rmnom(w[i], 1, z_rate)   #スイッチング変数を生成
      Zi11[doc_list[[i]]] <- z1[, 1]   #トピックに関係のある単語
      Zi12[doc_list[[i]]] <- (z1[, -1] * dir_list2[[i]]) %*% rep(1, length(dir_list1[[i]]))   #ディレクトリに関係のある単語
      
      #ディリクレ分布から混合率をサンプリング「
      z_freq <- as.numeric(t(z1) %*% doc_vec[[i]])
      gamma[[i]] <- as.numeric(extraDistr::rdirichlet(1, as.numeric(t(z1) %*% doc_vec[[i]]) + alpha1))
    }
  }
  #生成したスイッチング変数のインデックスを作成
  index_z11 <- which(Zi11==1)
  
  
  ##トピック分布からトピックをサンプリング
  #多項分布よりトピックをサンプリング
  Zi2 <- matrix(0, nrow=f, ncol=k)
  z_rate <- Lho1[index_z11, ] / Li1[index_z11]   #トピックの割当確率
  Zi2[index_z11, ] <- rmnom(length(index_z11), 1, z_rate)   #トピックをサンプリング
  Zi2_T <- t(Zi2)
  
  
  ##トピック分布のパラメータをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- Zi2_T[, doc_list[[i]], drop=FALSE] %*% doc_vec[[i]]
  }
  wsum <- wsum0 + alpha1   #ディリクレ分布のパラメータ
  theta <- extraDistr::rdirichlet(d, wsum)   #パラメータをサンプリング
  
  
  ##単語分布のパラメータをサンプリング
  #トピックの単語分布をサンプリング
  vsum0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vsum0[, j] <- Zi2_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]
  }
  vsum <- vsum0 + alpha2   
  phi1 <- extraDistr::rdirichlet(k, vsum)   #パラメータをサンプリング
  
  #ディレクトリの単語分布をサンプリング
  Zi0 <- Zi12[-index_z11]
  sparse_data0 <- sparse_data[-index_z11, ]
  dsum0 <- matrix(0, nrow=dir2, ncol=v)
  for(j in 1:dir2){
    dsum0[j, ] <- colSums(sparse_data0[Zi0==j, ]) 
  }
  dsum <- dsum0 + alpha2   #ディリクレ分布のパラメータ
  phi2 <- extraDistr::rdirichlet(dir2, dsum)   #パラメータをサンプリング
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi2
    THETA[, , mkeep] <- theta
  }  
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG0 <- matrix(0, nrow=f, ncol=dir2)
    for(i in 1:f){
      if(Zi12[i]==0) next
      SEG0[i, Zi12[i]] <- 1
    }
    SEG11 <- SEG11 + Zi11
    SEG12 <- SEG12 + SEG0
    SEG2 <- SEG2 + Zi2
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    Lho <- rep(0, d)
    for(i in 1:d){
      Lho[i] <- sum(log(rowSums(Lho_list[[i]])))
    }
    #サンプリング結果を確認
    print(rp)
    print(c(sum(Lho), LLst))
    print(mean(Zi11))
    print(round(cbind(phi2[, (v11-4):(v11+5)], phit2[, (v11-4):(v11+5)]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 2000/keep
RS <- R/keep

##サンプリング結果の可視化
#トピック分布の可視化
matplot(t(THETA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#単語分布の可視化
matplot(t(PHI1[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI1[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI1[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI1[7, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[4, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[6, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[8, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

##サンプリング結果の事後分布
#トピック分布の事後平均
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)
round(apply(THETA[, , burnin:RS], c(1, 2), sd), 3)

#単語分布の事後平均
round(cbind(t(apply(PHI1[, , burnin:RS], c(1, 2), mean)), t(phit1)), 3)
round(t(apply(PHI1[, , burnin:RS], c(1, 2), sd)), 3)
round(cbind(t(apply(PHI2[, , burnin:RS], c(1, 2), mean)), t(phit2)), 3)
round(t(apply(PHI2[, , burnin:RS], c(1, 2), sd)), 3)



##潜在変数のサンプリング結果の事後分布
seg11_rate <- SEG11 / max(SEG11); seg12_rate <- SEG12 / max(SEG11)
seg21_rate <- SEG12 / max(SEG11)
seg2_rate <- SEG2 / rowSums(SEG2)
seg11_rate[is.nan(seg11_rate)] <- 0; seg12_rate[is.nan(seg12_rate)] <- 0
seg21_rate[is.nan(seg21_rate)] <- 0
seg2_rate[is.nan(seg2_rate)] <- 0

#トピック割当結果を比較
round(cbind(SEG11, seg11_rate, ZT1), 3)
round(cbind(rowSums(SEG12), seg12_rate), 3)
round(cbind(rowSums(SEG2), seg2_rate, Z2), 3)


