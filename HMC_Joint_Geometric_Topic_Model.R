#####HMC Joint Geometric Topic Model####
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
w <- rtpois(hh, rgamma(item, 30.0, 0.225), a=1, b=Inf)   #訪問数
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
      if(length(index) > 0){
      cov <- runif(2, 0.005, 0.125) * diag(2)
      cov[1, 2] <- cov[2, 1] <- runif(1, -0.6, 0.6) * prod(sqrt(diag(cov)))
      geo_item0[index, ] <- mvrnorm(length(index), c(runif(1, longitude[1], longitude[2]), runif(1, latitude[1], latitude[2])), cov)
    }
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
  alpha2 <- 2.25
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


####ハミルトニアンモンテカルロ法でJoint Geometric Topic Modelを推定####
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
loglike <- function(phi, dt, d_par_matrix, d_id, hh, item, item_vec){
  #パラメータを設定
  phi_par <- exp(c(0, phi))
  
  #場所選択確率を設定
  denom_par <- (d_par_matrix * matrix(phi_par, nrow=hh, ncol=item, byrow=T))[d_id, ]   #分母を設定
  prob_spot <- denom_par / as.numeric(denom_par %*% item_vec)   #トピックごとに選択確率
  
  #完全データの対数尤度の和
  LL <- sum(log((dt * prob_spot) %*% item_vec))
  return(LL)
}

##場所選択確率のパラメータをサンプリングするための関数
#場所選択確率パラメータの対数事後分布の微分関数
dloglike <- function(phi, dt, d_par_matrix, d_id, hh, item, item_vec){
  #パラメータを設定
  phi_par <- exp(c(0, phi))
  
  #場所選択確率を設定
  denom_par <- (d_par_matrix * matrix(phi_par, nrow=hh, ncol=item, byrow=T))[d_id, ]   #分母を設定
  prob_spot <- denom_par / as.numeric(denom_par %*% item_vec)   #トピックごとに選択確率
  
  #勾配ベクトルを算出
  sc <- -colSums(dt - prob_spot)[-1]
  return(sc)
}

#場所選択確率パラメータのリープフロッグ法を解く関数
leapfrog <- function(r, z, D, e, L) {
  leapfrog.step <- function(r, z, e){
    r2 <- r  - e * D(z, dt, d_par_matrix, d_id0, hh, item, item_vec) / 2
    z2 <- z + e * r2
    r2 <- r2 - e * D(z2, dt, d_par_matrix, d_id0, hh, item, item_vec) / 2
    list(r=r2, z=z2) # 1回の移動後の運動量と座標
  }
  leapfrog.result <- list(r=r, z=z)
  for(i in 1:L) {
    leapfrog.result <- leapfrog.step(leapfrog.result$r, leapfrog.result$z, e)
  }
  leapfrog.result
}

##アルゴリズムの設定
R <- 1000
keep <- 2
burnin <- 200/keep
disp <- 5
LL1 <- -1000000000
iter <- 0
e <- 0.03
L <- 3

##インデックスとデータを設定
#インデックスの設定
v_index <- v_vec <- list()
d_vec <- sparseMatrix(sort(d_id), unlist(d_index), x=rep(1, f), dims=c(hh, f))

for(j in 1:item){
  v_index[[j]] <- which(v==j)
  v_vec[[j]] <- rep(1, length(v_index[[j]]))
}

#データの設定
Data <- as.matrix(sparse_data); storage.mode(Data) <- "integer"
d_par_matrix0 <- matrix(d0, nrow=hh, ncol=item, byrow=T)
item_vec <- rep(1, item)

##事前分布の設定
alpha01 <- 0.1
alpha02 <- rep(0, item-1)
inv_tau <- solve(100 * diag(item))

##パラメータの真値
beta <- betat 
theta <- thetat
phi <- phit

##初期値の設定
#パラメータの初期値
beta <- 1.0
theta <- extraDistr::rdirichlet(hh, rep(1.0, k))
phi <- cbind(0, mvrnorm(k, rep(0, item-1), 0.1 * diag(item-1)))

#場所の選択確率の初期値
phi_par <- t(exp(phi))
d_par <- exp(-beta/2 * d)
d_par_matrix <- matrix(exp(-beta/2 * d0), nrow=hh, ncol=item, byrow=T)
denom_par <- (d_par_matrix %*% phi_par)[d_id, ]   #分母を設定
prob_spot <- (phi_par[v, ] * d_par) / denom_par   #トピックごとに選択確率を算出

##パラメータの格納用配列
THETA <- array(0, dim=c(hh, k, R/keep))
PHI <- array(0, dim=c(k, item, R/keep))
SEG <- matrix(0, nrow=f, ncol=k)
gamma_rate <- rep(0, k)
storage.mode(SEG) <- "integer"


##対数尤度の基準値
#ユニグラムモデルの対数尤度
LLst <- sum(log(sparse_data %*% colSums(Data) / f))

#ベストな対数尤度を算出
#場所選択確率を更新
phi_par <- t(exp(phit))
d_par <- exp(-beta/2 * d)
d_par_matrix <- matrix(exp(-beta/2 * d0), nrow=hh, ncol=item, byrow=T)
denom_par <- (d_par_matrix %*% phi_par)[d_id, ]   #分母を設定
prob_spot <- (phi_par[v, ] * d_par) / denom_par   #トピックごとに選択確率を算出

#観測データの対数尤度
LLbest <- sum(log((thetat[d_id, ] * prob_spot) %*% rep(1, k)))   


####HMC法でパラメータをサンプリング####
for(rp in 1:R){   #dlがtol以上の場合は繰り返す
  
  ##トピックをサンプリング
  #トピックの選択確率を算出
  Lho <- theta[d_id, ] * prob_spot   #尤度関数
  prob_topic <- Lho / as.numeric(Lho %*% rep(1, k))   #トピック選択確率
  
  #多項分布からトピックをサンプリング
  Zi <- rmnom(f, 1, prob_topic)
  
  
  ##ユーザーごとにトピック分布のパラメータをサンプリング
  wsum <- d_vec %*% Zi + alpha01   #ディリレク分布のパラメータ
  theta <- extraDistr::rdirichlet(hh, wsum)   #ディリクレ分布からパラメータをサンプリング
  
  ##トピックごとに場所分布のパラメータをサンプリング
  for(j in 1:k){
    
    #トピックの割当を抽出
    index <- which(Zi[, j]==1)
    dt <- Data[index, ]
    d_id0 <- d_id[index]
    
    #HMCの新しいパラメータを生成
    rold <- rnorm(item-1)   #標準正規分布からパラメータを生成
    phid <- phi[j, -1]
    
    #リープフロッグ法による1ステップ移動
    res <- leapfrog(rold, phid, dloglike, e, L)
    rnew <- res$r
    phin <- res$z
    
    #移動前と移動後のハミルトニアン
    Hnew <- -(loglike(phin, dt, d_par_matrix, d_id0, hh, item, item_vec)) + sum(rnew^2)/2
    Hold <- -(loglike(phid, dt, d_par_matrix, d_id0, hh, item, item_vec)) + sum(rold^2)/2
    
    #HMC法によりパラメータの採択を決定
    rand <- runif(1)   #一様分布から乱数を発生
    gamma <- min(c(1, exp(Hold - Hnew)))   #採択率を決定
    gamma_rate[j] <- gamma
    
    #alphaの値に基づき新しいbetaを採択するかどうかを決定
    flag <- gamma > rand
    phi[j, -1] <- flag*phin + (1-flag)*phid
  }

  
  #場所選択確率を更新
  phi_par <- t(exp(phi))
  d_par <- exp(-beta/2 * d)
  d_par_matrix <- matrix(exp(-beta/2 * d0), nrow=hh, ncol=item, byrow=T)
  denom_par <- (d_par_matrix %*% phi_par)[d_id, ]   #分母を設定
  prob_spot <- (phi_par[v, ] * d_par) / denom_par   #トピックごとに選択確率を算出
  
  
  ##パラメータの格納とサンプリング結果の表示
  #パラメータを格納
  if(rp%%keep==0){
    #モデルのパラメータを格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #バーンイン期間を超えたらトピックを格納
    if(rp >= burnin){
      SEG <- SEG + Zi
    }
  }
  
  #対数尤度の計算とサンプリング結果を確認
  if(rp%%disp==0){
    #観測データの対数尤度
    LL <- sum(log((theta[d_id, ] * prob_spot) %*% rep(1, k)))   
    
    #サンプリング結果を表示
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(gamma_rate, 3))
  }
}

##トピックごとに場所分布のパラメータを推定


