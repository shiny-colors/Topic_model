#####連続空間トピックモデル#####
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

#set.seed(5723)

####データの発生####
k <- 10   #トピック数
d <- 2000   #ユーザー数
v <- 1000   #語彙数
w <- rpois(d, rgamma(d, 45, 0.3))   #1人あたりのページ閲覧数
f <- sum(w)   #総語彙数

#IDを設定
d_id <- rep(1:d, w)
t_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))


##単語がすべて生成されるまで繰り返す
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  ##パラメータの設定
  G0 <- GT0 <- extraDistr::rdirichlet(1, rep(2.0, v))   #単語出現率
  u <- ut <- mvrnorm(d, rep(0, k), diag(k))
  phi <- phit <- mvrnorm(v, rep(0, k), diag(k))
  
  ##データの生成
  word_list <- list()
  word_vec_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    #ディクレリ-多項分布から単語を生成
    alpha <- G0 * exp(u[i, ] %*% t(phi))
    words <- extraDistr::rdirmnom(w[i], 1, alpha)
    words_vec <- as.numeric(words %*% 1:v)
    
    #生成した単語を格納
    WX[i, ] <- colSums(words)
    word_list[[i]] <- words
    word_vec_list[[i]] <- words_vec
  }
  if(min(colSums(WX)) > 0){
    break
  }
}

#リストを変換
wd <- unlist(word_vec_list)
sparse_data <- as(WX, "CsparseMatrix")
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))   #単語ベクトルを行列化
word_dt <- t(word_data)
rm(word_list)

##インデックスを作成
dw_list <- list()
for(j in 1:v){
  index <- which(wd==j)
  dw_list[[j]] <- d_id[index]
}


####マルコフ連鎖モンテカルロ法で連続空間トピックモデルを推定####
##アルゴリズムの設定
R <- 2000
keep <- 2  
iter <- 0
burnin <- 300
disp <- 5

#データの設定
rej2 <- rep(0, v)
vec_v <- rep(1, v)
WX_vec <- as.numeric(WX)

#インデックスを設定
index_nzeros <- which(as.numeric(WX) > 0)
index_word <- list()
for(j in 1:v){
  index_word[[j]] <- which(WX[, j] > 0)
}

##事前分布の設定
cov <- diag(k)
inv_cov <- solve(cov)
mu <- rep(0, k)
sigma <- diag(k)


##パラメータの真値
G0 <- GT0
G0_data <- matrix(G0, nrow=d, ncol=v, byrow=T)
u <- ut
phi <- phit


##パラメータの初期値
G0 <- colSums(WX) / sum(WX)
G0_data <- matrix(G0, nrow=d, ncol=v, byrow=T)
u <- mvrnorm(d, rep(0, k), diag(k))
phi <- mvrnorm(v, rep(0, k), diag(k))


##パラメータの保存用配列
logl <- rep(0, R/keep)
U <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(v, k, R/keep))

##対数尤度の基準値
#ユニグラムモデルの対数尤度
par <- colSums(WX) / sum(WX)
LLst <- sum(WX %*% log(par))

#ベストな対数尤度
alpha <- matrix(GT0, nrow=d, ncol=v, byrow=T) * exp(ut %*% t(phit))
LLbest <- sum(lgamma(rowSums(alpha)) - lgamma(rowSums(alpha + WX))) + sum(lgamma(alpha + WX) - lgamma(alpha))


####マルコフ連鎖モンテカルロ法でパラメータをサンプリング####
for(rp in 1:R){
  
  ##ユーザートピック行列Uをサンプリング
  #新しいパラメータをサンプリング
  u_old <- u
  u_new <- u_old + mvrnorm(d, rep(0, k), diag(0.01, k))
  alphan <- matrix(G0, nrow=d, ncol=v, byrow=T) * exp(u_new %*% t(phi))
  alphad <- matrix(G0, nrow=d, ncol=v, byrow=T) * exp(u_old %*% t(phi))
  
  #Polya分布のパラメータを計算
  dirn_vec <- dird_vec <- rep(0, d*v)
  alphan_vec <- as.numeric(alphan)
  alphad_vec <- as.numeric(alphad)
  dirn_vec[index_nzeros] <- lgamma(alphan_vec[index_nzeros] + WX_vec[index_nzeros]) - lgamma(alphan_vec[index_nzeros])
  dird_vec[index_nzeros] <- lgamma(alphad_vec[index_nzeros] + WX_vec[index_nzeros]) - lgamma(alphad_vec[index_nzeros])
  
  #対数尤度と対数事前分布を計算
  dir_new <- lgamma(as.numeric(alphan %*% vec_v)) - lgamma(as.numeric((alphan + WX) %*% vec_v))
  dir_old <- lgamma(as.numeric(alphad %*% vec_v)) - lgamma(as.numeric((alphad + WX) %*% vec_v))
  lognew1 <- dir_new + as.numeric(matrix(dirn_vec, nrow=d, ncol=v) %*% vec_v)
  logold1 <- dir_old + as.numeric(matrix(dird_vec, nrow=d, ncol=v) %*% vec_v)
  logpnew1 <- -0.5 * rowSums(u_new %*% inv_cov * u_new)
  logpold1 <- -0.5 * rowSums(u_old %*% inv_cov * u_old)
  
  ##MHサンプリング
  rand <- runif(d)   #一様分布から乱数を発生
  LLind_diff <- exp(lognew1 + logpnew1 - logold1 - logpold1)   #採択率を計算
  tau <- (LLind_diff > 1)*1 + (LLind_diff <= 1)*LLind_diff
  
  #tauの値に基づき新しいbetaを採択するかどうかを決定
  flag <- matrix(((tau >= rand)*1 + (tau < rand)*0), nrow=d, ncol=k)
  rej1 <- mean(flag[, 1])
  u <- flag*u_new + (1-flag)*u_old   #alphaがrandを上回っていたら採択
  dir_vec <- flag[, 1]*dirn_vec + (1-flag[, 1])*dird_vec
  dir_mnd <- flag[, 1]*dir_new + (1-flag[, 1])*dir_old

  
  ##単語分布のパラメータphiをサンプリング
  #提案モデルのパラメータ
  er <- mvrnorm(v, rep(0, k), diag(0.01, k))   

  for(j in 1:v){
    index_vec <- index_word[[j]]
    
    #単語ごとにMHサンプリングを実行
    if(j==1){
      #新しいパラメータをサンプリング
      phid <- phi 
      alphan <- alphad <- G0_data * exp(u %*% t(phid))
      phin <- phid[j, ] + er[j, ]
      alphan[, j] <- G0[j] * exp(u %*% phin)
      
      #Polya分布のパラメータを更新
      dir_mnd1 <- dir_mnd
      dir_mnn2 <- dir_mnd2 <- matrix(dir_vec, nrow=d, ncol=v)
      dir_mnn1 <- lgamma(as.numeric(alphan %*% vec_v)) - lgamma(as.numeric((alphan + WX) %*% vec_v))
      dir_mnn2[index_vec, j] <- lgamma(alphan[index_vec, j] + WX[index_vec, j]) - lgamma(alphan[index_vec, j])
      
    } else {
      
      #新しいパラメータをサンプリング
      alphan <- alphad
      phin <- phi[j, ] + er[j, ]
      alphan[, j] <- G0[j] * exp(u %*% phin)
      
      #繰り返し2回目以降のパラメータを更新
      dir_mnn2 <- dir_mnd2
      dir_mnn1 <- lgamma(as.numeric(alphan %*% vec_v)) - lgamma(as.numeric((alphan + WX) %*% vec_v))
      dir_mnn2[index_vec, j] <- lgamma(alphan[index_vec, j] + WX[index_vec, j]) - lgamma(alphan[index_vec, j])
    }
  
    #対数尤度と対数事前分布を計算
    lognew2 <- sum(dir_mnn1) + sum(dir_mnn2)
    logold2 <- sum(dir_mnd1) + sum(dir_mnd2)
    logpnew2 <- lndMvn(phin, mu, cov)
    logpold2 <- lndMvn(phid[j, ], mu, cov)
  
    #MHサンプリング
    rand <- runif(1)   #一様分布から乱数を発生
    tau <- min(c(1, exp(lognew2 + logpnew2 - logold2 - logpold2)))   #採択率を計算

    #tauの値に基づき新しいbetaを採択するかどうかを決定
    rej2[j] <- flag <- as.numeric(tau >= rand)
    phi[j, ] <- flag*phin + (1-flag)*phid[j, ]
    alphad[, j] <- flag*alphan[, j] + (1-flag)*alphad[, j]
    dir_mnd1 <- flag*dir_mnn1 + (1-flag)*dir_mnd1
    dir_mnd2 <- flag*dir_mnn2 + (1-flag)*dir_mnd2
  }
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    logl[mkeep] <- lognew2
    U[, , mkeep] <- u
    PHI[, , mkeep] <- phi
  }
  
  #サンプリング結果を確認
  if(rp%%disp==0){
    print(rp)
    print(c(lognew2, LLbest, LLst))
    print(round(c(rej1, mean(rej2)), 3))
    print(round(cbind(u[1:5, ], ut[1:5, ]), 2))
    print(round(cbind(phi[1:5, ], phit[1:5, ]), 2))
  }
}

