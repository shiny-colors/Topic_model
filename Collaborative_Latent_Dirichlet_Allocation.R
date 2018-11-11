#####Collaborative Latent Dirichlet Allocation#####
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
k <- 20   #トピック数
hh <- 5000   #ユーザー数
item <- 2000   #アイテム数
g <- 1000   #語彙数
pt <- rtpois(hh, rgamma(hh, 20, 0.2), a=0, b=Inf)   #評価件数
hhpt <- sum(pt)   #総評価件数
w <- extraDistr::rtpois(item, rgamma(item, 30, 0.2), a=0, b=Inf)   #アイテムの単語数
f <- sum(w)   #総単語数


##IDとインデックスの設定
#IDの設定
user_id <- rep(1:hh, pt)
d_id <- rep(1:item, w)

#インデックスの設定
user_index <- d_index <- list()
vec <- c(0, cumsum(pt))
for(i in 1:hh){
  user_index[[i]] <- (1:hhpt)[(vec[i]+1):vec[i+1]]
}
vec <- c(0, cumsum(w))
for(j in 1:item){
  d_index[[j]] <- (1:f)[(vec[j]+1):vec[j+1]]
}

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

#アイテムの説明変数
k1 <- 2; k2 <- 4; k3 <- 4
v1 <- matrix(runif(item*k1, 0, 1), nrow=item, ncol=k1)
v2 <- matrix(0, nrow=item, ncol=k2)
for(j in 1:k2){
  pr <- runif(1, 0.25, 0.55)
  v2[, j] <- rbinom(item, 1, pr)
}
v3 <- rmnom(item, 1, runif(k3, 0.2, 1.25)); v3 <- v3[, -which.min(colSums(v3))]
v <- cbind(1, v1, v2, v3)   #データを結合

#パラメータ数
k1 <- ncol(x); k2 <- ncol(u); k3 <- ncol(v)


##アイテムの割当を生成
#セグメント割当を生成
topic <- 25
gamma <- extraDistr::rdirichlet(topic, rep(0.5, item))
z <- as.numeric(rmnom(hh, 1, extraDistr::rdirichlet(hh, rep(2.5, topic))) %*% 1:topic)

#多項分布からアイテムを生成
item_id_list <- list()
for(i in 1:hh){
  if(i%%100==0){
    print(i)
  }
  item_id_list[[i]] <- as.numeric(rmnom(pt[i], 1, gamma[z[user_id[user_index[[i]]]], ]) %*% 1:item)
}
item_id <- unlist(item_id_list)
item_index <- list()
for(j in 1:item){
  item_index[[j]] <- which(item_id==j)
}


##応答変数が妥当になるまでパラメータの生成を繰り返す
rp <- 0
repeat { 
  rp <- rp + 1
  
  ##LDAのパラメータを生成
  #LDAのディリクレ分布のパラメータを設定
  alpha01 <- rep(0.1, k)
  alpha02 <- matrix(0.015, nrow=k, ncol=g)
  for(j in 1:k){
    alpha02[j, matrix(1:item, nrow=k, ncol=g/k, byrow=T)[j, ]] <- 0.125
  }
  
  #ディリクレ分布からパラメータを生成
  theta <- thetat <- extraDistr::rdirichlet(item, alpha01)
  phi <- phit <- extraDistr::rdirichlet(k, alpha02)
  
  #アイテム出現確率が低いトピックを入れ替える
  index <- which(colMaxs(phi) < (k*5)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, alpha01)) %*% 1:k), index[j]] <- (k*5)/f
  }
  
  ##行列分解のパラメータを生成
  #素性ベクトルのパラメータ
  sigma <- sigmat <- 0.3
  beta <- betat <- c(5.5, rnorm(k1-1, 0, 0.5))
  
  #階層モデルの分散パラメータ
  Cov_u <- Cov_ut <- diag(runif(k, 0.005, 0.1), k)   #ユーザー-アイテムの階層モデルの分散
  Cov_v <- Cov_vt <- diag(runif(k, 0.005, 0.1), k)   #アイテムの階層モデルの分散
  
  #階層モデルの回帰係数を設定
  alpha_u <- alpha_ut <- matrix(rnorm(k*k2, 0, 0.3), nrow=k2, ncol=k)
  alpha_v <- alpha_vt <- matrix(rnorm(k*k3, 0, 0.3), nrow=k3, ncol=k)
  
  #行列分解のパラメータを生成
  theta_u <- theta_ut <- u %*% alpha_u + mvrnorm(hh, rep(0, k), Cov_u)
  theta_v <- theta_vt <- v %*% alpha_v + mvrnorm(item, rep(0, k), Cov_v)
  
  
  ##トピックと単語を生成
  #多項分布からトピックを生成
  Z <- rmnom(f, 1, theta[d_id, ])
  z_vec <- as.numeric(Z %*% 1:k)
  
  #トピックに基づき単語を生成
  word <- rmnom(f, 1, phi[z_vec, ])
  wd <- as.numeric(word %*% 1:g)
  
  
  ##正規分布から評価ベクトルを生成
  #評価スコアの期待値
  vec_topic <- rep(1, k)
  lambda <- as.numeric(x %*% beta)   #素性ベクトルの期待値
  uv <- as.numeric((theta_u[user_id, ] * (theta[item_id, ] + theta_v[item_id, ])) %*% vec_topic)   #行列分解のパラメータ
  mu <- lambda + uv   #期待値
  
  #評価ベクトルを生成
  y0 <- rnorm(hhpt, mu, sigma)
  
  #break条件
  print(sum(colSums(word)==0))
  print(c(max(y0), min(y0)))
  if(max(y0) < 15.0 & min(y0) > -4.0 & max(y0) > 12.0 & min(y0) < -2.0 & sum(colSums(word)==0)==0){
    break
  }
}

#生成したスコアを評価データに変換
y0_censor <- ifelse(y0 < 1, 1, ifelse(y0 > 10, 10, y0)) 
y <- round(y0_censor, 0)   #スコアを丸める

#スコア分布と単語分布
hist(colSums(word), col="grey", breaks=25, xlab="単語頻度", main="単語頻度分布")
hist(y0, col="grey", breaks=25, xlab="スコア", main="完全データのスコア分布")
hist(y, col="grey", breaks=25, xlab="スコア", main="観測されたスコア分布")

#スパース行列に変換
item_data <- sparseMatrix(1:hhpt, item_id, x=rep(1, hhpt), dims=c(hhpt, item))
item_data_T <- t(item_data)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, g))
word_data_T <- t(word_data)


####ギブスサンプリングでCollaborative Latent Dirichlet Allocationを推定####
##アイテムごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / rowSums(Bur)   #負担率
  r <- colSums(Br) / sum(Br)   #混合率
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##アルゴリズムの設定
R <- 3000
keep <- 2  
iter <- 0
burnin <- 500
disp <- 10

##事前分布の設定
#LDAのディリクレ分布の事前分布
alpha01 <- 0.1
alpha02 <- 0.1

#素性ベクトルの事前分布
delata <- rep(0, k1)
tau <- 100 * diag(k1)
inv_tau <- solve(tau)

#逆ガンマ分布の事前分布
s0 <- 1.0
v0 <- 1.0

#ユーザーの階層モデルの事前分布
Deltabar1 <- matrix(rep(0, k2*k), nrow=k2, ncol=k)   #階層モデルの回帰係数の事前分布の分散
ADelta1 <- 0.01 * diag(rep(1, k2))   #階層モデルの回帰係数の事前分布の分散
nu1 <- k2   #逆ウィシャート分布の自由度
V1 <- nu1 * diag(rep(1, k)) #逆ウィシャート分布のパラメータ

#アイテムの階層モデルの事前分布
Deltabar2 <- matrix(rep(0, k3*k), nrow=k3, ncol=k)   #階層モデルの回帰係数の事前分布の分散
ADelta2 <- 0.01 * diag(rep(1, k3))   #階層モデルの回帰係数の事前分布の分散
nu2 <- k3   #逆ウィシャート分布の自由度
V2 <- nu2 * diag(rep(1, k)) #逆ウィシャート分布のパラメータ

##パラメータの真値
#LDAのパラメータ
theta <- thetat
phi <- phit

#素性ベクトルのパラメータ
beta <- betat
sigma <- sigmat
lambda <- as.numeric(x %*% beta)

#階層モデルのパラメータ
alpha_u <- alpha_ut; Cov_u <- Cov_ut
mu_u <- u %*% alpha_u; inv_Cov_u <- solve(Cov_u)
alpha_v <- alpha_vt; Cov_v <- Cov_vt
mu_v <- v %*% alpha_v; inv_Cov_v <- solve(Cov_v)

#行列分解のパラメータ
theta_u <- theta_ut
theta_v <- theta_vt
uv <- as.numeric((theta_u[user_id, ] * (theta[item_id, ] + theta_v[item_id, ])) %*% vec_topic)


##パラメータの初期値を設定
#LDAのパラメータ
theta <- extraDistr::rdirichlet(item, rep(2.0, k))
phi <- extraDistr::rdirichlet(k, rep(2.0, g))

#素性ベクトルのパラメータ
beta <- as.numeric(solve(t(x) %*% x) %*% t(x) %*% y)
sigma <- 0.5
lambda <- as.numeric(x %*% beta)

#階層モデルのパラメータ
alpha_u <- matrix(0, nrow=k2, ncol=k); Cov_u <- 0.01 * diag(k)
mu_u <- u %*% alpha_u; inv_Cov_u <- solve(Cov_u)
alpha_v <- matrix(0, nrow=k3, ncol=k); Cov_v <- 0.01 * diag(k)
mu_v <- v %*% alpha_v; inv_Cov_v <- solve(Cov_v)

#行列分解のパラメータ
theta_u <- mu_u + mvrnorm(hh, rep(0, k), Cov_u)
theta_v <- mu_v + mvrnorm(item, rep(0, k), Cov_v)
uv <- as.numeric((theta_u[user_id, ] * (theta[item_id, ] + theta_v[item_id, ])) %*% vec_topic)

##パラメータの格納用配列
#LDAの格納用配列
THETA <- array(0, dim=c(item, k, R/keep))
PHI <- array(0, dim=c(k, g, R/keep))
SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG) <- "integer"

#行列分解の格納用配列
BETA <- matrix(0, nrow=R/keep, ncol=k1)
SIGMA <- rep(0, R/keep)
ALPHA_U <- array(0, dim=c(k2, k, R/keep))
ALPHA_V <- array(0, dim=c(k3, k, R/keep))
COV_U <- array(0, dim=c(k, k, R/keep))
COV_V <- array(0, dim=c(k, k, R/keep))
THETA_U <- array(0, dim=c(hh, k, R/keep))
THETA_V <- array(0, dim=c(item, k, R/keep))


##データとインデックスの設定
#データの設定
xx <- t(x) %*% x + inv_tau; inv_xx <- solve(xx)

#インデックスの設定
ui_id <- iu_id <- list()
for(i in 1:hh){
  ui_id[[i]] <- item_id[user_index[[i]]]
}
for(j in 1:item){
  iu_id[[j]] <- user_id[item_index[[j]]]
}

user_data <- sparseMatrix(1:hhpt, user_id, x=rep(1, hhpt), dims=c(hhpt, hh))
user_data_T <- t(user_data)
d_data <- sparseMatrix(1:f, d_id, x=rep(1, f), dims=c(f, item))
d_data_T <- t(d_data)


##対数尤度の基準値を設定
#LDAの対数尤度の基準値
LLst1 <- sum(d_data %*% log(colSums(d_data)/f))   #1パラメータの対数尤度
LLbest1 <- sum(log((thetat[d_id, ] * t(phit)[wd, ]) %*% vec_topic))   #ベストな対数尤度

##行列分解モデルの対数尤度の基準値
LLst2 <- sum(dnorm(y, mean(y), sd(y), log=TRUE))   #1パラメータモデルの対数尤度
mu <- as.numeric(x %*% betat) + as.numeric((theta_ut[user_id, ] * (thetat[item_id, ] + theta_vt[item_id, ])) %*% vec_topic)
LLbest2 <- sum(dnorm(y, mu, sigmat, log=TRUE))   #ベストな対数尤度



####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##LDAのパラメータをサンプリング
  #多項分布からトピックをサンプリング
  Lho <- theta[d_id, ] * t(phi)[wd, ]   #トピックの尤度
  prob <- Lho / as.numeric(Lho %*% vec_topic)   #トピックの割当確率
  Zi <- rmnom(f, 1, prob)   #トピックをサンプリング
  z_vec <- as.numeric(Zi %*% 1:k)
  
  #ディリクレ分布からトピック分布をサンプリング
  dsums <- d_data_T %*% Zi + alpha01   #ディリクレ分布のパラメータ
  theta <- extraDistr::rdirichlet(item, dsums)   #パラメータをサンプリング
  
  #ディリクレ分布から単語分布をサンプリング
  wsums <- t(word_data_T %*% Zi) + alpha02   #ディリクレ分布のパラメータ
  phi <- extraDistr::rdirichlet(k, wsums)   #パラメータをサンプリング
  
  
  ##素性ベクトルのパラメータをサンプリング
  #応答変数の生成
  y_er <- y - uv   #モデル誤差
  
  #多変量正規分布のパラメータを設定
  xy <- t(x) %*% y_er
  mu_vec <- as.numeric(inv_xx %*% xy)   #多変量正規分布の平均ベクトル
  
  #多変量正規分布から素性ベクトルをサンプリング
  beta <- mvrnorm(1, mu_vec, sigma^2*inv_xx)
  lambda <- as.numeric(x %*% beta)
  
  ##モデルの標準偏差をサンプリング
  #逆ガンマ分布のパラメータ
  er <- y - lambda - uv   #モデルの誤差
  s1 <- as.numeric(t(er) %*% er) + s0
  v1 <- hhpt + v0
  
  #逆ガンマ分布から標準偏差をサンプリング
  sigma <- sqrt(1/rgamma(1, v1/2, s1/2))
  
  
  ##ユーザーの特徴行列をサンプリング
  #モデルの応答変数
  y_er <- y - lambda
  
  for(i in 1:hh){
    #特徴ベクトルの事後分布のパラメータ
    X <- theta_v[ui_id[[i]], ] + theta[ui_id[[i]], ]
    Xy <- t(X) %*% y_er[user_index[[i]]]
    inv_XXV <- solve(t(X) %*% X + inv_Cov_u)
    theta_vec <- inv_XXV %*% (Xy + inv_Cov_u %*% mu_u[i, ])
    
    #多変量正規分布からユーザー特徴ベクトルをサンプリング
    theta_u[i, ] <- mvrnorm(1, theta_vec, sigma^2*inv_XXV)
  }
  
  ##アイテムの特徴行列をサンプリング
  #モデルの応答変数
  y_er <- y - lambda - as.numeric((theta_u[user_id, ]*theta[item_id, ]) %*% vec_topic)
  
  for(j in 1:item){
    #特徴ベクトルの事後分布のパラメータ
    X <- theta_u[iu_id[[j]], ]
    Xy <- t(X) %*% y_er[item_index[[j]]]
    inv_XXV <- solve(t(X) %*% X + inv_Cov_v)
    theta_vec <- inv_XXV %*% (Xy + inv_Cov_v %*% mu_v[j, ])
    
    #多変量正規分布からユーザー特徴ベクトルをサンプリング
    theta_v[j, ] <- mvrnorm(1, theta_vec, sigma^2*inv_XXV)
  } 
  uv <- as.numeric((theta_u[user_id, ] * (theta[item_id, ] + theta_v[item_id, ])) %*% vec_topic)   #行列分解のパラメータを更新
  
  
  ##階層モデルのパラメータをサンプリング
  #ユーザーの行列分解のパラメータをサンプリング
  out_u <- rmultireg(theta_u, u, Deltabar1, ADelta1, nu1, V1)
  alpha_u <- out_u$B
  Cov_u <- diag(diag(out_u$Sigma))
  mu_u <- u %*% alpha_u
  inv_Cov_u <- solve(Cov_u)
  
  #アイテムの行列分解のパラメータをサンプリング
  out_v <- rmultireg(theta_v, v, Deltabar2, ADelta2, nu2, V2)
  alpha_v <- out_v$B
  Cov_v <- diag(diag(out_v$Sigma))
  mu_v <- v %*% alpha_v
  inv_Cov_v <- solve(Cov_v)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    mkeep <- rp/keep
    #LDAのサンプリング結果の格納
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #行列分解のサンプリング結果の格納
    BETA[mkeep, ] <- beta
    THETA_U[, , mkeep] <- theta_u
    THETA_V[, , mkeep] <- theta_v
    ALPHA_U[, , mkeep] <- alpha_u
    ALPHA_V[, , mkeep] <- alpha_v 
    COV_U[, , mkeep] <- Cov_u
    COV_V[, , mkeep] <- Cov_v
  }
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG <- SEG + Zi
  }
  
  if(rp%%disp==0){
    #LDAの対数尤度
    LL1 <- sum(log((theta[d_id, ] * t(phi)[wd, ]) %*% vec_topic)) 
    
    #行列分解モデルの対数尤度
    mu <- as.numeric(x %*% beta) + as.numeric((theta_u[user_id, ] * (theta[item_id, ] + theta_v[item_id, ])) %*% vec_topic)
    LL2 <- sum(dnorm(y, mu, sigma, log=TRUE))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL1, LLbest1, LLst1))
    print(c(LL2, LLbest2, LLst2))
  }
}

