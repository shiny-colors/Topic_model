#####Factorized Latent Dirichlet Allocation#####
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
##データの設定
k <- 15   #トピック数
hh <- 5000   #ユーザー数
item <- 2500   #アイテム数
v <- 1000   #語彙数 
w <- rpois(item, rgamma(item, 60, 0.4))   #1文書あたりの語彙数
pt <- rtpois(hh, rgamma(hh, 25.0, 0.225), a=1, b=Inf)   #購買接触数
f <- sum(w)   #総語彙数
hhpt <- sum(pt)   #総スコア数
vec_k <- rep(1, k)

#IDの設定
d_id <- rep(1:item, w)   #文書ID
no_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))
user_id <- rep(1:hh, pt)   #ユーザーID
t_id <- as.numeric(unlist(tapply(1:hhpt, user_id, rank)))
user_list <- list()
for(i in 1:hh){
  user_list[[i]] <- which(user_id==i)
}

##アイテムの割当を生成
#セグメント割当を生成
topic <- 25
phi <- extraDistr::rdirichlet(topic, rep(0.5, item))
z <- as.numeric(rmnom(hh, 1,  extraDistr::rdirichlet(hh, rep(2.5, topic))) %*% 1:topic)

#多項分布からアイテムを生成
item_id_list <- list()
for(i in 1:hh){
  if(i%%100==0){
    print(i)
  }
  item_id_list[[i]] <- as.numeric(rmnom(pt[i], 1, phi[z[user_id[user_list[[i]]]], ]) %*% 1:item)
}
item_id <- unlist(item_id_list)
item_list <- list(); item_n <- rep(0, item)
for(j in 1:item){
  item_list[[j]] <- which(item_id==j)
  item_n[j] <- length(item_list[[j]])   
}

#スパース行列を作成
user_data <- sparseMatrix(1:hhpt, user_id, x=rep(1, hhpt), dims=c(hhpt, hh))
user_data_T <- t(user_data)
item_data <- sparseMatrix(1:hhpt, item_id, x=rep(1, hhpt), dims=c(hhpt, item))
item_data_T <- t(item_data)

#生成したデータを可視化
freq_item <- plyr::count(item_id); freq_item$x <- as.character(freq_item$x)
hist(freq_item$freq, breaks=25, col="grey", xlab="アイテムの購買頻度", main="アイテムの購買頻度分布")
gc(); gc()


##素性ベクトルを生成
k1 <- 3; k2 <- 5; k3 <- 5
x1 <- matrix(runif(hhpt*k1, 0, 1), nrow=hhpt, ncol=k1)
x2 <- matrix(0, nrow=hhpt, ncol=k2)
for(j in 1:k2){
  pr <- runif(1, 0.25, 0.55)
  x2[, j] <- rbinom(hhpt, 1, pr)
}
x3 <- rmnom(hhpt, 1, runif(k3, 0.2, 1.25)); x3 <- x3[, -which.min(colSums(x3))]
x <- cbind(1, x1, x2, x3)   #データを結合
col_x <- ncol(x)

##階層モデルの説明変数を生成
#ユーザーの説明変数
k1 <- 2; k2 <- 4; k3 <- 5
u1 <- matrix(runif(hh*k1, 0, 1), nrow=hh, ncol=k1)
u2 <- matrix(0, nrow=hh, ncol=k2)
for(j in 1:k2){
  pr <- runif(1, 0.25, 0.55)
  u2[, j] <- rbinom(hh, 1, pr)
}
u3 <- rmnom(hh, 1, runif(k3, 0.2, 1.25)); u3 <- u3[, -which.min(colSums(u3))]
u <- cbind(1, u1, u2, u3)   #データを結合
col_u <- ncol(u)

#アイテムの説明変数
k1 <- 2; k2 <- 4; k3 <- 4
g1 <- matrix(runif(item*k1, 0, 1), nrow=item, ncol=k1)
g2 <- matrix(0, nrow=item, ncol=k2)
for(j in 1:k2){
  pr <- runif(1, 0.25, 0.55)
  g2[, j] <- rbinom(item, 1, pr)
}
g3 <- rmnom(item, 1, runif(k3, 0.2, 1.25)); g3 <- g3[, -which.min(colSums(g3))]
g <- cbind(1, g1, g2, g3)   #データを結合
col_g <- ncol(g)


####トピックモデルのデータを生成####
##パラメータを設定
#ディリクレ事前分布のパラメータ
alpha1 <- rep(0.15, k)
alpha2 <- rep(0.05, v)

#全種類の単語が出現するまで繰り返す
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  #ディリクレ分布からパラメータを生成
  theta <- thetat <- extraDistr::rdirichlet(item, alpha1)
  phi <- extraDistr::rdirichlet(k, alpha2)
  
  #単語出現確率が低いトピックを入れ替える
  index <- which(colMaxs(phi) < (k*10)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(2.0, k))) %*% 1:k), index[j]] <- (k*10)/f
  }
  phit <- phi

  ##トピックと単語を生成
  word_list <- Z_list <- list()
  WX <- matrix(0, nrow=item, ncol=v)
  
  for(i in 1:item){
    #トピックを生成
    z <- rmnom(w[i], 1, theta[i, ])
    z_vec <- as.numeric(z %*% 1:k)
    
    #単語を生成
    word <- rmnom(w[i], 1, phi[z_vec, ])
    WX[i, ] <- colSums(word)
    
    #データを格納
    word_list[[i]] <- as.numeric(word %*% 1:v)
    Z_list[[i]] <- z
  }
  if(min(colSums(WX)) > 0) break
}

#リストを変換
wd <- unlist(word_list)
Z <- do.call(rbind, Z_list)

#スパース行列を作成
d_data <- sparseMatrix(1:f, d_id, x=rep(1, f), dims=c(f, item))
d_data_T <- t(d_data)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
word_data_T <- t(word_data)

#インデックスを設定
doc_list <- wd_list <- list()
for(i in 1:item){
  doc_list[[i]] <- which(d_id==i)
}
for(j in 1:v){
  wd_list[[j]] <- which(wd==j)
}

#トピック分布の期待値を設定
Z_score <- matrix(0, nrow=item, ncol=k) 
for(i in 1:item){
  Z_score[i, ] <- colMeans(Z[doc_list[[i]], ])
}


####評価ベクトルを生成####
rp <- 0
repeat { 
  rp <- rp + 1
  
  ##パラメータを設定
  #素性ベクトルのパラメータ
  sigma <- sigmat <- 0.5
  beta <- betat <- c(5.5, rnorm(col_x-1, 0, 0.75))
  
  #階層モデルの分散パラメータ
  Cov_u <- Cov_ut <- runif(1, 0.1, 0.4)   #ユーザー-アイテムの階層モデルの標準偏差
  Cov_v <- Cov_vt <- runif(1, 0.1, 0.4)   #アイテムの階層モデルの標準偏差
  Cov_z <- Cov_zt <- diag(runif(k, 0.01, 0.1), k)   #トピックのユーザー特徴ベクトルの階層モデルの分散
  
  #階層モデルの回帰係数を設定
  alpha_u <- alpha_ut <- rnorm(col_u, 0, 0.35)
  alpha_v <- alpha_vt <- rnorm(col_g, 0, 0.35)
  alpha_z <- alpha_zt <- mvrnorm(col_u, rep(0, k), runif(k, 0.1, 0.25) * diag(k))
  
  #変量効果と特徴ベクトルのパラメータを生成
  theta_u <- theta_ut <- u %*% alpha_u + rnorm(hh, 0, Cov_u)
  theta_v <- theta_vt <- g %*% alpha_v + rnorm(item, 0, Cov_v)
  theta_z <- theta_zt <- u %*% alpha_z + mvrnorm(hh, rep(0, k), Cov_z)
  
  #正規分布からスコアを生成
  mu <- as.numeric(x %*% beta + theta_u[user_id] + theta_v[item_id] + (Z_score[item_id, ] * theta_z[user_id, ]) %*% vec_k)
  y0 <- rnorm(hhpt, mu, sigma)
  
  #break条件
  print(round(c(max(y0), min(y0)), 3))
  if(max(y0) < 15.0 & min(y0) > -4.0 & max(y0) > 11.0 & min(y0) < -1.0){
    break
  }
}

#生成したスコアを評価データに変換
y0_censor <- ifelse(y0 < 1, 1, ifelse(y0 > 10, 10, y0)) 
y <- round(y0_censor, 0)   #スコアを丸める

#スコア分布と単語分布
hist(y0, col="grey", breaks=25, xlab="スコア", main="完全データのスコア分布")
hist(y, col="grey", breaks=25, xlab="スコア", main="観測されたスコア分布")


####マルコフ連鎖モンテカルロ法でfLDAを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / as.numeric(Bur %*% vec_k)   #負担率
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}

##アルゴリズムの設定
R <- 3000   #サンプリング回数
keep <- 2   #2回に1回の割合でサンプリング結果を格納
disp <- 10
iter <- 0
burnin <- 1000/keep

##インデックスとデータの設定
#インデックスを設定
index_u <- index_w <- y_vec <- z_dt <- list()
for(j in 1:item){
  index_y <- item_list[[j]]; n <- item_n[j]
  index_u[[j]] <- rep(user_id[index_y], w[j]); index_w[[j]] <- rep(1:w[j], length(index_y))   #インデックス
  y_vec[[j]] <- rep(y[index_y], w[j])   #スコアのベクトル
  z_dt[[j]] <- sparseMatrix(rep(1:w[j], rep(n, w[j])), 1:(n*w[j]), x=rep(1, n*w[j]), dims=c(w[j], n*w[j]))   #和を取るための疎行列
}

#データの設定
xx <- t(x) %*% x
uu <- t(u) %*% u
gg <- t(g) %*% g


##事前分布の設定
#トピックモデルの事前分布
alpha01 <- 0.1
alpha02 <- 0.1

#行列分解の事前分布
beta01 <- 0
beta02 <- 0
s0 <- 0.1
v0 <- 0.1
Cov_x <- 100 * diag(col_x); inv_Cov_x <- solve(Cov_x)
tau_u <- 100 * diag(ncol(u)); inv_tau_u <- solve(tau_u)
tau_v <- 100 * diag(ncol(g)); inv_tau_g <- solve(tau_v)
Deltabar <- matrix(0, nrow=ncol(u), ncol=k)   #階層モデルの回帰係数の事前分布の平均
ADelta <- 0.01 * diag(1, ncol(u))   #階層モデルの回帰係数の事前分布の分散
nu <- k + 1   #逆ウィシャート分布の自由度
V <- nu * diag(rep(1, k)) #逆ウィシャート分布のパラメータ


##パラメータの真値
#トピックモデルのパラメータ
theta <- thetat
phi <- phit
wsum <- as.matrix(d_data_T %*% Z)
Z_score <- wsum / w

#素性ベクトルのパラメータ
sigma <- sigmat
beta <- betat
beta_mu <- as.numeric(x %*% beta)

#階層モデルの分散パラメータ
Cov_u <- Cov_ut; inv_Cov_u <- 1 / Cov_u 
Cov_v <- Cov_vt; inv_Cov_v <- 1 / Cov_v 
Cov_z <- Cov_zt; inv_Cov_z <- solve(Cov_z)

#階層モデルの回帰係数を設定
alpha_u <- alpha_ut; u_mu <- as.numeric(u %*% alpha_u)
alpha_v <- alpha_vt; v_mu <- as.numeric(g %*% alpha_v)
alpha_z <- alpha_zt; z_mu <- u %*% alpha_z

#変量効果と特徴ベクトルのパラメータ
theta_u <- theta_ut; theta_user <- theta_u[user_id]
theta_v <- theta_vt; theta_item <- theta_v[item_id]
theta_z <- theta_zt; theta_topic <- theta_z[user_id, ]


##初期値の設定
#トピックモデルの初期値
theta <- extraDistr::rdirichlet(item, rep(1.0, k))
phi <- extraDistr::rdirichlet(k, rep(1.0, v))
Zi <- rmnom(f, 1, rep(1/k, k))
wsum <- as.matrix(d_data_T %*% Z)
Z_score <- wsum / w

#素性ベクトルのパラメータ
sigma <- 1.0
beta <- as.numeric(solve(t(x) %*% x) %*% t(x) %*% y)
beta_mu <- as.numeric(x %*% beta)

#階層モデルの分散パラメータ
Cov_u <- 0.2; inv_Cov_u <- 1 / Cov_u 
Cov_v <- 0.2; inv_Cov_v <- 1 / Cov_v
Cov_z <- 0.01 * diag(k); inv_Cov_z <- solve(Cov_z)

#階層モデルの回帰係数を設定
alpha_u <- rnorm(col_u, 0, 0.1); u_mu <- as.numeric(u %*% alpha_u)
alpha_v <- rnorm(col_g, 0, 0.1); v_mu <- as.numeric(g %*% alpha_v)
alpha_z <- mvrnorm(col_u, rep(0, k), 0.01 * diag(k)); z_mu <- u %*% alpha_z

#変量効果と特徴ベクトルのパラメータを生成
theta_u <- u %*% alpha_u + rnorm(hh, 0, Cov_u); theta_user <- theta_u[user_id]
theta_v <- g %*% alpha_v + rnorm(item, 0, Cov_v); theta_item <- theta_v[item_id]
theta_z <- u %*% alpha_z + mvrnorm(hh, rep(0, k), Cov_z); theta_topic <- theta_z[user_id, ]


##パラメータの格納用配列
#トピックモデルのパラメータの格納用配列
THETA <- array(0, dim=c(item, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG) <- "integer"

#モデルパラメータの格納用配列
d <- 0
BETA <- matrix(0, nrow=R/keep, ncol=ncol(x))
SIGMA <- rep(0, R/keep)
THETA_U <- matrix(0, nrow=R/keep, ncol=hh)
THETA_V <- matrix(0, nrow=R/keep, ncol=item)
THETA_Z <- array(0, dim=c(hh, k, R/keep))

#階層モデルの格納用配列
ALPHA_U <- matrix(0, nrow=R/keep, ncol=col_u)
ALPHA_V <- matrix(0, nrow=R/keep, ncol=col_g)
ALPHA_Z <- array(0, dim=c(col_u, k, R/keep))
COV_U <- COV_V <- rep(0, R/keep)
COV_Z <- array(0, dim=c(k, k, R/keep))

##対数尤度の基準値
#1パラメータモデル
LLst <- sum(dnorm(y, mean(y), sd(y), log=TRUE))

#ベストモデルの対数尤度
score <- as.matrix(d_data_T %*% Z / w)
uz <- rowSums(score[item_id, ] * theta_zt[user_id, ])
mu <- as.numeric(x %*% betat + theta_ut[user_id] + theta_vt[item_id] + uz)
LLbest <- sum(dnorm(y, mu, sigmat, log=TRUE))


####ギブスサンプリングムでパラメータをサンプリング####
for(rp in 1:R){
  
  ##トピックごとの評価スコアの尤度を設定
  #トピック因子を除いたスコアの期待値
  mu_uv <- as.numeric(x %*% beta + theta_user + theta_item)   #トピック因子を除いた期待値
  wsum_z <- (wsum - rmnom(item, 1, wsum / as.numeric(wsum %*% vec_k)))   #トピック分布に比例してトピックを除外
  
  #評価スコアのトピック割当ごとの対数尤度
  LLi <- matrix(0, nrow=item, ncol=k)   #対数尤度の格納用配列
  for(j in 1:k){
    #トピックスコアの割当
    wsum_z0 <- wsum_z; wsum_z0[, j] <- wsum_z0[, j] + 1
    z_score <- (wsum_z0 / w)[item_id, ]
    
    #対数尤度を計算
    uz <- as.numeric((z_score * theta_topic) %*% vec_k)
    LLi[, j] <- as.numeric(item_data_T %*% dnorm(y, mu_uv + uz, sigma, log=TRUE))
  }
  
  ##単語トピックをサンプリング
  #単語ごとにトピックの割当確率を設定
  Lho <- theta[d_id, ] * t(phi)[wd, ] * exp(LLi - rowMaxs(LLi))[d_id, ]   #トピック割当ごとの尤度
  topic_rate <- Lho / as.numeric(Lho %*% vec_k)   #トピックの割当確率
  
  #多項分布からトピックをサンプリング
  Zi <- rmnom(f, 1, topic_rate)
  z_vec <- as.numeric(Zi %*% 1:k)
  
  
  ##トピックモデルのパラメータを推定
  #トピック分布をサンプリング
  wsum <- as.matrix(d_data_T %*% Zi)
  theta <- extraDistr::rdirichlet(item, wsum + alpha01)
  Z_score <- wsum / w; z_score <- Z_score[item_id, ]
  
  #単語分布のサンプリング
  vsum <- as.matrix(t(word_data_T %*% Zi)) + alpha02
  phi <- extraDistr::rdirichlet(k, vsum)
  
  
  ##素性ベクトルのパラメータをサンプリング
  #応答変数の設定
  uz <- as.numeric((z_score * theta_topic) %*% vec_k)
  y_er <- y - theta_u[user_id] - theta_v[item_id] - uz
  
  #素性ベクトルの事後分布のパラメータ
  Xy <- t(x) %*% y_er
  inv_XXV <- solve(xx + inv_Cov_x)
  mu_vec <- inv_XXV %*% Xy   #事後分布の平均
  
  #多変量正規分布から素性ベクトルをサンプリング
  beta <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  beta_mu <- as.numeric(x %*% beta)
  
  ##モデルの標準偏差をサンプリング
  #逆ガンマ分布のパラメータ
  er <- y - beta_mu - theta_user - theta_item - uz   #モデルの誤差
  s1 <- as.numeric(t(er) %*% er) + s0
  v1 <- hhpt + v0
  
  #逆ガンマ分布から標準偏差をサンプリング
  sigma <- sqrt(1/rgamma(1, v1/2, s1/2))
  
  
  ##ユーザの変量効果をサンプリング
  #ユーザー変量効果の応答変数の設定
  y_er <- y - beta_mu - theta_item - uz
  
  for(i in 1:hh){
    #ユーザー変量効果の事後分布のパラメータ
    w_omega <- 1/Cov_u^2 + pt[i]/sigma^2
    weight <- (1/Cov_u^2) / w_omega
    mu_scalar <- weight*u_mu[i] + (1-weight)*mean(y_er[user_list[[i]]])
  
    #正規分布からパラメータサンプリング
    theta_u[i] <- rnorm(1, mu_scalar, 1/ w_omega)
  }
  theta_user <- theta_u[user_id]
  
  ##アイテムの変量効果をサンプリング
  #アイテム変量効果の応答変数の設定
  y_er <- y - beta_mu - theta_user - uz
  
  for(j in 1:item){
    #アイテム変量効果の事後分布のパラメータ
    w_omega <- 1/Cov_v^2 + item_n[j]/sigma^2
    weight <- (1/Cov_v^2) / w_omega
    mu_scalar <- weight*v_mu[j] + (1-weight)*mean(y_er[item_list[[j]]])
    
    #正規分布からユーザー変量効果をサンプリング
    theta_v[j] <- rnorm(1, mu_scalar, 1/ w_omega)
  }
  theta_item <- theta_v[item_id]
  
  
  ##ユーザーごとに特徴ベクトルをサンプリング
  #応答変数の設定
  y_er <- y - beta_mu - theta_user - theta_item
  
  for(i in 1:hh){
    #特徴ベクトルの事後分布のパラメータ
    index <- user_list[[i]]   #ユーザーインデックス
    X <- z_score[index, ]   #トピックの経験分布
    Xy <- t(X) %*% y_er[index]
    inv_XXV <- solve(t(X) %*% X + inv_Cov_z)
    mu_vec <- inv_XXV %*% (Xy + inv_Cov_z %*% z_mu[i, ])   #事後分布の平均
    
    #多変量正規分布から特徴ベクトルをサンプリング
    theta_z[i, ] <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  }
  theta_topic <- theta_z[user_id, ]
  uz <- as.numeric((z_score * theta_topic) %*% vec_k)
  
  
  ##ユーザーの変量効果の階層モデルのパラメータをサンプリング
  #事後分布のパラメータ
  Xy <- t(u) %*% theta_u
  inv_XXV <- solve(t(u) %*% u + inv_tau_u)
  mu_vec <- inv_XXV %*% Xy   #事後分布の平均
  
  #多変量正規分布から素性ベクトルをサンプリング
  alpha_u <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  u_mu <- as.numeric(u %*% alpha_u)
  
  #モデルの標準偏差をサンプリング
  #逆ガンマ分布のパラメータ
  er <- theta_u - u_mu   #モデルの誤差
  s1 <- as.numeric(t(er) %*% er) + s0
  v1 <- hh + v0
  
  #逆ガンマ分布から標準偏差をサンプリング
  Cov_u <- sqrt(1/rgamma(1, v1/2, s1/2))
  
  
  ##アイテムの変量効果の階層モデルのパラメータをサンプリング
  #事後分布のパラメータ
  Xy <- t(g) %*% theta_v
  inv_XXV <- solve(t(g) %*% g + inv_tau_g)
  mu_vec <- inv_XXV %*% Xy   #事後分布の平均
  
  #多変量正規分布から素性ベクトルをサンプリング
  alpha_v <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  v_mu <- as.numeric(g %*% alpha_v)
  
  #モデルの標準偏差をサンプリング
  #逆ガンマ分布のパラメータ
  er <- theta_v - v_mu   #モデルの誤差
  s1 <- as.numeric(t(er) %*% er) + s0
  v1 <- item + v0
  
  #逆ガンマ分布から標準偏差をサンプリング
  Cov_v <- sqrt(1/rgamma(1, v1/2, s1/2))
  
  
  ##ユーザー特徴行列の階層モデルのパラメータを推定
  #多変量回帰モデルからパラメータをサンプリング
  out <- rmultireg(theta_z, u, Deltabar, ADelta, nu, V)
  alpha_z <- out$B; z_mu <- u %*% alpha_z   
  Cov_z <- out$Sigma; inv_Cov_z <- solve(Cov_z)
  
  ##パラメータの格納とサンプリング結果の格納
  if(rp%%keep==0){
    mkeep <- rp/keep
    #トピックモデルのパラメータを格納
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #モデルパラメータの格納
    BETA[mkeep, ] <- beta
    SIGMA[mkeep] <- sigma
    THETA_U[mkeep, ] <- theta_u
    THETA_V[mkeep, ] <- theta_v
    THETA_Z[, , mkeep] <- theta_z
    
    #階層モデルの格納
    ALPHA_U[mkeep, ] <- alpha_u
    ALPHA_V[mkeep, ] <- alpha_v
    ALPHA_Z[, , mkeep] <- alpha_z 
    COV_U[mkeep] <- Cov_u
    COV_V[mkeep] <- Cov_v
    COV_Z[, , mkeep] <- Cov_z

    if(rp >= burnin){
      d <- d + 1
      SEG <- SEG + Zi
    }
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    mu <- beta_mu + theta_user + theta_item + uz   #期待値
    LL <- sum(dnorm(y, mu, sigma, log=TRUE))   #対数尤度の和 
    
    #サンプリング結果の表示
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(rbind(beta, betat), 3))
  }
}

