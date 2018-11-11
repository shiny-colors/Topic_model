#####Joint Geometric Topic Model####
options(warn=0)
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(data.table)
library(bayesm)
library(HMM)
library(stringr)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)
#set.seed(2506787)

####データの発生####
##データの発生
k <- 20   #トピック数
hh <- 2000   #ユーザー数
item <- 1000   #場所数
w <- rtpois(hh, rgamma(item, 25, 0.25), a=1, b=Inf)   #訪問数
f <- sum(w)   #総訪問数

##IDとインデックスの設定
#IDの設定
d_id <- rep(1:hh, w)
t_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))
geo_id01 <- rep(1:hh, rep(item, hh))
geo_id02 <- rep(1:item, hh)

#インデックスの設定
d_index <- geo_index <- list()
for(i in 1:hh){
  d_index[[i]] <- which(d_id==i)
}
for(i in 1:hh){
  geo_index[[i]] <- which(geo_id01==i)
}

##すべてのアイテムが生成されるまで繰り返す
for(rp in 1:1000){
  print(rp)
  
  ##ユーザーとアイテムの経緯度を生成
  #ユーザーの場所集合を生成
  s <- 30 
  rate <- extraDistr::rdirichlet(1, rep(2.0, s))
  point <- as.numeric(rmnom(hh, 1, rate) %*% 1:s)
  
  #経緯度を生成
  longitude <- c(0, 5); latitude <- c(0, 5)
  geo_user0 <- matrix(0, nrow=hh, ncol=2)
  for(j in 1:s){
    index <- which(point==j)
    cov <- runif(2, 0.01, 0.15) * diag(2)
    cov[1, 2] <- cov[2, 1] <- runif(1, -0.6, 0.6) * prod(sqrt(diag(cov)))
    geo_user0[index, ] <- mvrnorm(length(index), c(runif(1, longitude[1], longitude[2]), runif(1, latitude[1], latitude[2])), cov)
  }
  geo_user <- min(geo_user0) + geo_user0
  plot(geo_user, xlab="経度", ylab="緯度", main="ユーザーの場所集合の分布") 
  
  
  #スポットの場所集合を生成
  s <- 25
  rate <- extraDistr::rdirichlet(1, rep(2.0, s))
  point <- as.numeric(rmnom(item, 1, rate) %*% 1:s)
  
  #経緯度を生成
  longitude <- c(0, 5); latitude <- c(0, 5)
  geo_item0 <- matrix(0, nrow=item, ncol=2)
  for(j in 1:s){
    index <- which(point==j)
    cov <- runif(2, 0.005, 0.125) * diag(2)
    cov[1, 2] <- cov[2, 1] <- runif(1, -0.6, 0.6) * prod(sqrt(diag(cov)))
    geo_item0[index, ] <- mvrnorm(length(index), c(runif(1, longitude[1], longitude[2]), runif(1, latitude[1], latitude[2])), cov)
  }
  geo_item <- min(geo_item0) + geo_item0
  plot(geo_item, xlab="経度", ylab="緯度", main="スポットの分布") 
  
  
  #ユーザーと場所のユークリッド距離
  d0 <- sqrt(rowSums((geo_user[geo_id01, ] - geo_item[geo_id02, ])^2))
  hist(d0, breaks=50, xlab="ユークリッド距離", main="2地点間のユークリッド距離の分布", col="grey")
  matrix(d0, nrow=hh, ncol=item, byrow=T)
  
  ##パラメータを生成
  #トピック分布を生成
  alpha1 <- rep(0.1, k)
  theta <- thetat <- extraDistr::rdirichlet(hh, alpha1)
  
  #場所分布の生成
  alpha2 <- 2.0
  beta <- betat <- 1.0   #バンド幅のパラメータ
  phi <- phit <- cbind(0, mvrnorm(k, rep(0, item-1), alpha2^2*diag(item-1)))
  
  
  ##応答変数を生成
  Z_list <- V_list <- d_list <- prob_list <- list()
  VX <- matrix(0, nrow=hh, ncol=item); storage.mode(VX) <- "integer"
  
  for(i in 1:hh){
    #トピックを生成
    z <- rmnom(w[i], 1, theta[i, ])
    z_vec <- as.numeric(z %*% 1:k)
    
    #訪問確率を決定
    par <- exp(phi[z_vec, ]) * matrix(exp(-beta/2 * d0[geo_index[[i]]]), nrow=w[i], ncol=item, byrow=T)
    prob <- par / rowSums(par)
    
    #訪問した場所を生成
    v <- rmnom(w[i], 1, prob)
    v_vec <- as.numeric(v %*% 1:item)
    
    #データを格納
    d_list[[i]] <- d0[geo_index[[i]]][v_vec]
    prob_list[[i]] <- rowSums(prob * v)  
    Z_list[[i]] <- z
    V_list[[i]] <- v_vec
    VX[i, ] <- colSums(v)
  }
  #break条件
  if(min(colSums(VX)) > 0) break
}

#リストを変換
d <- unlist(d_list)
prob <- unlist(prob_list)
Z <- do.call(rbind, Z_list); storage.mode(Z) <- "integer"
v <- unlist(V_list)
sparse_data <- sparseMatrix(i=1:f, j=v, x=rep(1, f), dims=c(f, item))
sparse_data_T <- t(sparse_data)


#データの可視化
plot(geo_user, xlab="経度", ylab="緯度", main="ユーザーの場所集合の分布") 
plot(geo_item, xlab="経度", ylab="緯度", main="スポットの分布")
hist(d0, breaks=50, xlab="ユークリッド距離", main="2地点間のユークリッド距離の分布", col="grey")


####EMアルゴリズムでJoint Geometric Topic Modelを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / rowSums(Bur)   #負担率
  r <- colSums(Br) / sum(Br)   #混合率
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##完全データの対数尤度の和を算出する関数
loglike <- function(x, v, Data, theta, prob_topic, d_par_matrix, d_id, j){
  #パラメータを設定
  phi_par <- exp(c(0, x))
  
  #場所選択確率を設定
  denom_par <- (d_par_matrix %*% phi_par)[d_id, ]   #分母を設定
  prob_spot <- (phi_par[v] * d_par) / denom_par   #トピックごとに選択確率
  
  #完全データの対数尤度の和
  LL <- sum(prob_topic[, j] * log(theta[d_id, j] * prob_spot))
  return(LL)
}

gradient <- function(x, v, Data, theta, prob_topic, d_par_matrix, d_id, j){
  #パラメータを設定
  phi_par <- exp(c(0, x))
  
  #場所選択確率を設定
  denom_par <- (d_par_matrix0 * matrix(phi_par, nrow=hh, ncol=item, byrow=T))   #分母を設定
  prob_spot <- denom_par[d_id, ] / rowSums(denom_par)[d_id]   #トピックごとに選択確率
  
  #勾配ベクトルを算出
  sc <- colSums(prob_topic[, j] * (Data - prob_spot))[-1]
  return(sc)
}

##インデックスとデータを設定
#インデックスの設定
v_index <- v_vec <- d_vec <- list()
for(i in 1:hh){
  d_vec[[i]] <- rep(1, length(d_index[[i]]))
}
for(j in 1:item){
  v_index[[j]] <- which(v==j)
  v_vec[[j]] <- rep(1, length(v_index[[j]]))
}

#データの設定
Data <- as.matrix(sparse_data); storage.mode(Data) <- "integer"
d_par_matrix0 <- matrix(d0, nrow=hh, ncol=item, byrow=T)
  
##パラメータの真値
beta <- betat 
theta <- thetat
phi <- phit

##パラメータの初期値
beta <- 1.0
theta <- extraDistr::rdirichlet(hh, rep(1.0, k))
phi <- cbind(0, mvrnorm(k, rep(0, item-1), 0.1 * diag(item-1)))

##場所の選択確率の初期値
#パラメータを設定
phi_par <- t(exp(phi))
d_par <- exp(-beta/2 * d)
d_par_matrix <- matrix(exp(-beta/2 * d0), nrow=hh, ncol=item, byrow=T)
denom_par <- (d_par_matrix %*% phi_par)[d_id, ]   #分母を設定

#トピックごとに選択確率を算出
prob_spot <- (phi_par[v, ] * d_par) / denom_par


##更新ステータス
LL1 <- -1000000000
dl <- 100   #EMステップでの対数尤度の差の初期値
tol <- 10.0
iter <- 0 

####EMアルゴリズムでパラメータを推定####
while(abs(dl) >= tol){ #dlがtol以上の場合は繰り返す
  
  ##Eステップでトピック選択確率を算出
  Lho <- theta[d_id, ] * prob_spot   #尤度関数
  prob_topic <- Lho / as.numeric(Lho %*% rep(1, k))   #トピック選択確率
  prob_topic_T <- t(prob_topic)
  

  ##Mステップでトピック分布のパラメータを推定
  #ユーザーごとにトピック割当を設定
  wsum <- matrix(0, nrow=hh, ncol=k) 
  for(i in 1:hh){
    wsum[i, ] <- prob_topic_T[, d_index[[i]], drop=FALSE] %*% d_vec[[i]]
  }
  #トピック分布を更新
  theta <- wsum / w   
  
  ##準ニュートン法で場所分布のパラメータを推定
  #トピックごとにパラメータを更新
  for(j in 1:k){
    x <- phi[j, -1]
    res <- optim(x, loglike, gr=gradient, v, Data, theta, prob_topic, d_par_matrix, d_id, j, 
                 method="BFGS", hessian=FALSE, control=list(fnscale=-1, trace=FALSE, maxit=1))
    phi[j, -1] <- res$par
  }
  
  #場所選択確率を更新
  phi_par <- t(exp(phi))
  d_par <- exp(-beta/2 * d)
  d_par_matrix <- matrix(exp(-beta/2 * d0), nrow=hh, ncol=item, byrow=T)
  denom_par <- (d_par_matrix %*% phi_par)[d_id, ]   #分母を設定
  prob_spot <- (phi_par[v, ] * d_par) / denom_par   #トピックごとに選択確率を算出
  
  
  ##対数尤度を更新
  LL <- sum(log((theta[d_id, ] * prob_spot) %*% rep(1, k)))   #観測データの対数尤度
  iter <- iter + 1
  dl <- LL - LL1
  LL1 <- LL
  print(LL)
  gc()
}


