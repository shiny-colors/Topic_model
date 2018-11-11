#####Binomial Latent Dirichlet Allocation model#####
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
r <- 10   #評価スコア数
k1 <- 7   #ユーザートピック数
k2 <- 8   #アイテムトピック数
K <- matrix(1:(k1*k2), nrow=k1, ncol=k2, byrow=T)   #トピックの配列
hh <- 3000   #レビュアー数
item <- 1000   #アイテム数

##IDと欠損ベクトルの作成
#IDを仮設定
user_id0 <- rep(1:hh, rep(item, hh))
item_id0 <- rep(1:item, hh)

#欠損ベクトルを作成
for(rp in 1:100){
  m_vec <- rep(0, hh*item)
  for(i in 1:item){
    prob <- rbeta(1, 8.0, 50.0)
    m_vec[item_id0==i] <- rbinom(hh, 1, prob)
  }
  m_index <- which(m_vec==1)
  
  #完全なIDを設定
  user_id <- user_id0[m_index]
  item_id <- item_id0[m_index]
  d <- length(user_id)   #総レビュー数
  
  #すべてのパターンが生成されればbreak
  if(length(unique(user_id))==hh & length(unique(item_id))==item) break
}

##パラメータの設定
#ディリクレ分布の事前分布を設定
alpha11 <- rep(0.2, k1)
alpha12 <- rep(0.2, k2)
alpha2 <- rep(0.15, r)

#ディリクレ分布からパラメータを生成
theta1 <- thetat1 <- extraDistr::rdirichlet(hh, alpha11)
theta2 <- thetat2 <- extraDistr::rdirichlet(item, alpha12)
phi <- phit <- extraDistr::rdirichlet(k1*k2, alpha2)

#モデルに基づきデータを生成
y <- matrix(0, nrow=d, ncol=r)
y_vec <- rep(0, d)
Z1 <- matrix(0, nrow=d, ncol=k1)
Z2 <- matrix(0, nrow=d, ncol=k2)

for(i in 1:d){
  if(i%%10000==0){
    print(i)
  }
  #データインデックスを抽出
  u_index <- user_id[i]
  i_index <- item_id[i]
  
  #トピックを生成
  z1 <- as.numeric(rmnom(1, 1, theta1[u_index, ]))
  z2 <- as.numeric(rmnom(1, 1, theta2[i_index, ]))
  
  #トピックに基づき評価スコアを生成
  k_index <- K[which.max(z1), which.max(z2)]
  y[i, ] <- rmnom(1, 1, phi[k_index, ])
  y_vec[i] <- which.max(y[i, ])
  
  #データを格納
  Z1[i, ] <- z1
  Z2[i, ] <- z2
}

####マルコフ連鎖モンテカルロ法でBi LDAを推定####
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
burnin <- 1000
disp <- 10

##事前分布の設定
alpha1 <- 0.25
alpha2 <- 0.25
beta <- 0.25

##パラメータの真値
theta1 <- thetat1
theta2 <- thetat2
phi <- phit

##パラメータの初期値を設定
#トピック分布の初期値
theta1 <- extraDistr::rdirichlet(hh, rep(1.0, k1))
theta2 <- extraDistr::rdirichlet(item, rep(1.0, k2))
phi <- extraDistr::rdirichlet(k1*k2, rep(1.0, r)) 

##パラメータの格納用配列
THETA1 <- array(0, dim=c(hh, k1, R/keep))
THETA2 <- array(0, dim=c(item, k2, R/keep))
PHI <- array(0, dim=c(k1*k2, r, R/keep))
SEG1 <- matrix(0, nrow=d, ncol=k1)
SEG2 <- matrix(0, nrow=d, ncol=k2)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##データとインデックスの設定
#インデックスの設定
user_index <- user_vec <- list()
item_index <- item_vec <- list()
y_list <- y_ones <- list()
for(i in 1:hh){
  user_index[[i]] <- which(user_id==i)
  user_vec[[i]] <- rep(1, length(user_index[[i]]))
}
for(j in 1:item){
  item_index[[j]] <- which(item_id==j)
  item_vec[[j]] <- rep(1, length(item_index[[j]]))
}
for(j in 1:r){
  y_list[[j]] <- which(y_vec==j)
  y_ones[[j]] <- rep(1, length(y_list[[j]]))
}

#データの設定
r_vec <- rep(1, r)
k1_vec <- rep(1, k2)
k2_vec <- rep(1, k1)
index_k1 <- rep(1:k1, rep(k2, k1))
index_k2 <- rep(1:k2, k1)

##対数尤度の基準値と最良値
#基準の対数尤度
par <- colSums(y) / d
LLst <- sum(y %*% log(par))

#最良の対数尤度
phi11 <- t(phit)[y_vec, ] * thetat2[item_id, index_k2]
par_u0 <- matrix(0, nrow=d, ncol=k1)
for(j in 1:k1){
  par_u0[, j] <- phi11[, K[j, ]] %*% k1_vec
}
par_u1 <- thetat1[user_id, ] * par_u0   #ユーザートピックの期待尤度
LLbest <- sum(log(rowSums(par_u1)))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##ユーザートピックをサンプリング
  #ユーザートピックの条件付き確率
  phi11 <- t(phi)[y_vec, ] * theta2[item_id, index_k2]
  par_u0 <- matrix(0, nrow=d, ncol=k1)
  for(j in 1:k1){
    par_u0[, j] <- phi11[, K[j, ]] %*% k1_vec
  }
  par_u1 <- theta1[user_id, ] * par_u0   #ユーザートピックの期待尤度
  
  #潜在変数の割当確率からトピックをサンプリング
  z1_rate <- par_u1 / as.numeric(par_u1 %*% k2_vec)
  Zi1 <- rmnom(d, 1, z1_rate)
  Zi1_T <- t(Zi1)
  
  
  ##アイテムトピックをサンプリング
  #アイテムトピックの条件付き確率
  phi12 <- t(phi)[y_vec, ] * theta1[user_id, index_k1]
  par_i0 <- matrix(0, nrow=d, ncol=k2)
  for(j in 1:k2){
    par_i0[, j] <- phi12[, K[, j]] %*% k2_vec
  }
  par_i1 <- theta2[item_id, ] * par_i0   #ユーザートピックの期待尤度
  
  #潜在変数の割当確率からトピックをサンプリング
  z2_rate <- par_i1 / as.numeric(par_i1 %*% k1_vec)
  Zi2 <- rmnom(d, 1, z2_rate)
  Zi2_T <- t(Zi2)
  
  #ユーザーとアイテムトピックを統合
  Zi <- Zi1[, index_k1] * Zi2[, index_k2]
  Zi_T <- t(Zi)
  
  
  ##パラメータをサンプリング
  #ユーザーのトピック分布をサンプリング
  wusum0 <- matrix(0, nrow=d, ncol=k1)
  for(i in 1:hh){
    wusum0[i, ] <- Zi1_T[, user_index[[i]]] %*% user_vec[[i]]
  }
  wusum <- wusum0 + alpha1   #ディリクレ分布のパラメータ
  theta1 <- extraDistr::rdirichlet(hh, wusum)   #ディリクレ分布からtheta11をサンプリング
  
  #アイテムのトピック分布をサンプリング
  wisum0 <- matrix(0, nrow=item, ncol=k2)
  for(i in 1:item){
    wisum0[i, ] <- Zi2_T[, item_index[[i]]] %*% item_vec[[i]]
  }
  wisum <- wisum0 + alpha2   #ディリクレ分布のパラメータ
  theta2 <- extraDistr::rdirichlet(item, wisum)   #ディリクレ分布からtheta11をサンプリング
  
  
  ##評価スコア分布をサンプリング
  vsum0 <- matrix(0, nrow=k1*k2, ncol=r)
  for(j in 1:r){
    vsum0[, j] <- Zi_T[, y_list[[j]]] %*% y_ones[[j]]
  }
  vsum <- vsum0 + beta   #ディリクレ分布のパラメータ
  phi <- extraDistr::rdirichlet(k1*k2, vsum)   #ディリクレ分布からetaをサンプリング
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    PHI[, , mkeep] <- phi
  }  
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG1 <- SEG1 + Zi1
    SEG2 <- SEG2 + Zi2
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL <- sum(log(rowSums(par_u1)))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(cbind(theta1[1:10, ], thetat1[1:10, ]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep
RS <- R/keep

##サンプリング結果のプロット
#評価スコアのトピック分布のサンプリング結果をプロット
matplot(t(THETA1[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[250, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[500, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[50, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[150, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[200, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#評価スコアのトピック分布のサンプリング結果をプロット
matplot(t(THETA21[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[250, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[500, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[50, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[150, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[200, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#スコア分布のサンプリング結果の可視化
matplot(t(ETA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[15, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[20, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#単語分布のサンプリング結果の可視化
matplot(t(PHI[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[7, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[9, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[4, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[8, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[12, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[15, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[4, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

##サンプリング結果の要約
#サンプリング結果の事後平均
round(cbind(apply(LAMBDA[, , burnin:RS], c(1, 2), mean), lambdat), 3)   #スイッチング変数の混合率の事後平均
round(cbind(apply(THETA11[, , burnin:RS], c(1, 2), mean), thetat11), 3)   #ユーザーのトピック割当の事後平均
round(cbind(apply(THETA12[, , burnin:RS], c(1, 2), mean), thetat12), 3)   #アイテムのトピック割当の事後平均
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phit)), 3)   #ユーザーの単語分布の事後平均
round(cbind(t(apply(GAMMA[, , burnin:RS], c(1, 2), mean)), t(gammat)), 3)   #アイテムの単語分布の事後平均
round(cbind(t(apply(OMEGA[, , burnin:RS], c(1, 2), mean)), t(omegat)), 3)   #評価スコアの単語分布の事後平均

#サンプリング結果の事後信用区間
round(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(apply(THETA1[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(THETA2[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)
round(t(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)
round(t(apply(OMEGA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(OMEGA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)


##サンプリングされた潜在変数の要約
n <- max(SEG1)
round(cbind(SEG1/n, Z1), 3)
round(cbind(rowSums(SEG21), SEG21/max(rowSums(SEG21)), Z21 %*% 1:k21), 3)
round(cbind(rowSums(SEG22), SEG22/max(rowSums(SEG22)), Z22 %*% 1:k22), 3)

