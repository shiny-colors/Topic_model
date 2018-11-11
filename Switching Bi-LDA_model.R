#####Switching Binomial LDAモデル#####
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
k1 <- 10   #ユーザートピック
k2 <- 15   #アイテムトピック
hh <- 1000   #レビュアー数
item <- 200   #アイテム数
v1 <- 500   #ユーザートピックの語彙数
v2 <- 500   #アイテムトピックの語彙数
v <- v1 + v2   #総語彙数
v1_index <- 1:v1
v2_index <- (v1+1):v

##IDと欠損ベクトルの作成
#IDを仮設定
user_id0 <- rep(1:hh, rep(item, hh))
item_id0 <- rep(1:item, hh)

#欠損ベクトルを作成
for(rp in 1:100){
  m_vec <- rep(0, hh*item)
  for(i in 1:item){
    prob <- runif(1, 0.025, 0.16)
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

#単語数を設定
w <- rpois(d, rgamma(d, 25, 0.5))   #文書あたりの単語数
f <- sum(w)   #総単語数
n_user <- plyr::count(user_id)$freq
n_item <- plyr::count(item_id)$freq

#単語IDを設定
u_id <- rep(user_id, w)
i_id <- rep(item_id, w)

#インデックスを設定
user_index <- list()
item_index <- list()
for(i in 1:hh){
  user_index[[i]] <- which(user_id==i)
}
for(j in 1:item){
  item_index[[j]] <- which(item_id==j)
}


##パラメータの設定
#ディリクレ分布の事前分布の設定
alpha11 <- rep(0.15, k1)
alpha12 <- rep(0.15, k2)
alpha21 <- c(rep(0.1, v1), rep(0.0025, v2))
alpha22 <- c(rep(0.0025, v1), rep(0.1, v2))
beta1 <- 12.5; beta2 <- 15.0

##すべての単語が出現するまでデータの生成を続ける
for(rp in 1:1000){

  #ディリクレ分布からパラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(hh, alpha11)
  theta2 <- thetat2 <- extraDistr::rdirichlet(item, alpha12)
  phi <- phit <- extraDistr::rdirichlet(k1, alpha21)
  gamma <- gammat <- extraDistr::rdirichlet(k2, alpha22)
  lambda <- lambdat <- rbeta(hh, beta1, beta2)
  
  ##モデルに基づきデータを生成
  WX <- matrix(0, nrow=d, ncol=v)
  Z1_list <- Z21_list <- Z22_list <- wd_list <- list()
  
  for(i in 1:d){
    if(i%%1000==0){
      print(i)
    }
    #ユーザーとアイテムを抽出
    u_index <- user_id[i]
    i_index <- item_id[i]
    
    #ベルヌーイ分布からスイッチング変数を生成
    z1 <- rbinom(w[i], 1, lambda[u_index])
    index_z1 <- which(z1==1)
    
    #ユーザートピックを生成
    z21 <- matrix(0, nrow=w[i], ncol=k1)
    if(sum(z1) > 0){
      z21[index_z1, ] <- rmnom(sum(z1), 1, theta1[u_index, ])
    }
    z21_vec <- as.numeric(z21 %*% 1:k1)
    
    #アイテムトピックを生成
    z22 <- matrix(0, nrow=w[i], ncol=k2)
    if(sum(1-z1) > 0){
      z22[-index_z1, ] <- rmnom(sum(1-z1), 1, theta2[i_index, ])
    }
    z22_vec <- as.numeric(z22 %*% 1:k2)
    
    #トピックから単語を生成
    words <- matrix(0, nrow=w[i], ncol=v)
    if(sum(z1) > 0){
      words[index_z1, ] <- rmnom(sum(z1), 1, phi[z21_vec[index_z1], ])
    }
    if(sum(1-z1) > 0){
      words[-index_z1, ] <- rmnom(sum(1-z1), 1, gamma[z22_vec[-index_z1], ])
    }
    word_vec <- as.numeric(words %*% 1:v)
    WX[i, ] <- colSums(words)
    
    #データを格納
    wd_list[[i]] <- word_vec
    Z1_list[[i]] <- z1
    Z21_list[[i]] <- z21
    Z22_list[[i]] <- z22
  }
  
  #すべての単語が生成できたらbreak
  if(min(colSums(WX)) > 0) break
}

#リストを変換
wd <- unlist(wd_list)
Z1 <- unlist(Z1_list); rt <- mean(Z1)
Z21 <- do.call(rbind, Z21_list)
Z22 <- do.call(rbind, Z22_list)
storage.mode(Z21) <- "integer"
storage.mode(Z22) <- "integer"
storage.mode(WX) <- "integer"


####マルコフ連鎖モンテカルロ法でSwitching Binomial LDAを推定####
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
alpha11 <- 0.25; alpha12 <- 0.25
alpha21 <- 0.1; alpha22 <- 0.1
beta01 <- 1.0; beta02 <- 1.0

##パラメータの真値
theta1 <- thetat1 
theta2 <- thetat2
phi <- phit
gamma <- gammat
lambda <- lambdat

##初期値の設定
#トピック分布の初期値
theta1 <- extraDistr::rdirichlet(hh, rep(1.0, k1))
theta2 <- extraDistr::rdirichlet(item, rep(1.0, k2))

#ユーザーの単語分布の初期値
u_word <- as.matrix(data.frame(id=user_id, WX) %>%
                      dplyr::group_by(id) %>%
                      dplyr::summarise_all(funs(sum)))[, 2:(v+1)]
phi <- extraDistr::rdirichlet(k1, apply(u_word, 2, sd))

#アイテムの単語分布の初期値
i_word <- as.matrix(data.frame(id=item_id, WX) %>%
                      dplyr::group_by(id) %>%
                      dplyr::summarise_all(funs(sum)))[, 2:(v+1)]
gamma <- extraDistr::rdirichlet(k2, apply(i_word, 2, sd)/10)

#混合率の初期値
lambda <- rep(0.5, hh)


##パラメータの格納用配列
THETA1 <- array(0, dim=c(hh, k1, R/keep))
THETA2 <- array(0, dim=c(item, k2, R/keep))
PHI <- array(0, dim=c(k1, v, R/keep))
GAMMA <- array(0, dim=c(k2, v, R/keep))
LAMBDA <- matrix(0, nrow=R/keep, ncol=hh)
SEG1 <- rep(0, f)
SEG21 <- matrix(0, nrow=f, ncol=k1)
SEG22 <- matrix(0, nrow=f, ncol=k2)
storage.mode(SEG21) <- "integer"
storage.mode(SEG22) <- "integer"

##インデックスとデータの設定
#インデックスの設定
user_list <- user_vec <- list()
item_list <- item_vec <- list()
wd_list <- wd_vec <- list()
user_n <- rep(0, hh)
item_n <- rep(0, item)
for(i in 1:hh){
  user_list[[i]] <- which(u_id==i)
  user_vec[[i]] <- rep(1, length(user_list[[i]]))
  user_n[i] <- sum(user_vec[[i]])
}
for(i in 1:item){
  item_list[[i]] <- which(i_id==i)
  item_vec[[i]] <- rep(1, length(item_list[[i]]))
  item_n[i] <- sum(item_vec[[i]])
}
for(j in 1:v){
  wd_list[[j]] <- which(wd==j)
  wd_vec[[j]] <- rep(1, length(wd_list[[j]]))
}

#データの設定
vec1 <- rep(1, k1)
vec2 <- rep(1, k2)

##対数尤度の基準値
par <- colSums(WX) / sum(WX)
LLst <- sum(WX %*% log(par))


####ギブスサンプラーでパラメータをサンプリング####
for(rp in 1:R){
  
  ##ベルヌーイ分布よりスイッチング変数をサンプリング
  #ユーザーおよびアイテムの期待尤度を設定
  Li_user <- theta1[u_id, ] * t(phi)[wd, ]   #ユーザー尤度
  par_user <- as.numeric(Li_user %*% vec1)   #ユーザーの期待尤度
  Li_item <- theta2[i_id, ] * t(gamma)[wd, ]   #アイテム尤度
  par_item <- as.numeric(Li_item %*% vec2)   #アイテムの期待尤度
  
  #潜在確率からスイッチング変数を生成
  r <- lambda[u_id]   #スイッチング変数の事前分布
  par_user_r <- r * par_user
  par_item_r <- (1-r) * par_item
  s_prob <- par_user_r / (par_user_r + par_item_r)   #スイッチング変数の割当確率
  Zi1 <- rbinom(f, 1, s_prob)   #ベルヌーイ分布からスイッチング変数を生成
  index_z1 <- which(Zi1==1)
  
  #ベータ分布から混合率をサンプリング
  for(i in 1:hh){
    n <- sum(Zi1[user_list[[i]]])
    beta1 <- n + beta01
    beta2 <- user_n[i] - n + beta02
    lambda[i] <- rbeta(1, beta1, beta2)   #ベータ分布からlambdaをサンプリング
  }
  
  
  ##ユーザーおよびアイテムのトピックをサンプリング
  #トピックの割当確率を推定
  z_rate1 <- Li_user[index_z1, ] / par_user[index_z1]   #ユーザーのトピック割当確率
  z_rate2 <- Li_item[-index_z1, ] / par_item[-index_z1]   #アイテムのトピック割当確率
  
  #多項分布からトピックを生成
  Zi21 <- matrix(0, nrow=f, ncol=k1)
  Zi22 <- matrix(0, nrow=f, ncol=k2)
  Zi21[index_z1, ] <- rmnom(nrow(z_rate1), 1, z_rate1)
  Zi22[-index_z1, ] <- rmnom(nrow(z_rate2), 2, z_rate2)
  Zi21_T <- t(Zi21)
  Zi22_T <- t(Zi22)
  
  
  ##トピックモデルのパラメータをサンプリング
  #ユーザーのトピック分布をサンプリング
  wusum0 <- matrix(0, nrow=hh, ncol=k1)
  for(i in 1:hh){
    wusum0[i, ] <- Zi21_T[, user_list[[i]]] %*% user_vec[[i]]
  }
  wusum <- wusum0 + alpha11   #ディリクレ分布のパラメータ
  theta1 <- extraDistr::rdirichlet(hh, wusum)   #ディリクレ分布からtheta1をサンプリング
  
  #アイテムのトピック分布をサンプリング
  wisum0 <- matrix(0, nrow=item, ncol=k2)
  for(i in 1:item){
    wisum0[i, ] <- Zi22_T[, item_list[[i]]] %*% item_vec[[i]]
  }
  wisum <- wisum0 + alpha12   #ディリクレ分布のパラメータ
  theta2 <- extraDistr::rdirichlet(item, wisum)   #ディリクレ分布からtheta2をサンプリング
  
  
  #ユーザーおよびアイテムの単語分布をサンプリング
  vusum0 <- matrix(0, nrow=k1, ncol=v)
  visum0 <- matrix(0, nrow=k2, ncol=v)
  for(j in 1:v){
    vusum0[, j] <- Zi21_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]
    visum0[, j] <- Zi22_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]
  }
  vusum <- vusum0 + alpha21; visum <- visum0 + alpha22   #ディリクレ分布のパラメータ
  phi <- extraDistr::rdirichlet(k1, vusum)   #ディリクレ分布から単語分布をサンプリング
  gamma <- extraDistr::rdirichlet(k2, visum)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma
    LAMBDA[mkeep, ] <- lambda
  }  
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG1 <- SEG1 + Zi1
    SEG21 <- SEG21 + Zi21
    SEG22 <- SEG22 + Zi22
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL <- sum(log(Zi1*par_user + (1-Zi1)*par_item))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL, LLst))
    print(c(mean(Zi1), rt))
    print(round(cbind(phi[, (v1-4):(v1+5)], phit[, (v1-4):(v1+5)]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep
RS <- R/keep

##サンプリング結果のプロット
#スイッチング変数の混合率の可視化
matplot(LAMBDA[, 1:25], type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(LAMBDA[, 101:125], type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(LAMBDA[, 501:525], type="l", xlab="サンプリング回数", ylab="パラメータ")

#トピック分布のサンプリング結果をプロット
matplot(t(THETA1[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[250, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[500, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[50, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[150, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[200, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

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



##サンプリング結果の要約
#サンプリング結果の事後平均
round(cbind(colMeans(LAMBDA[burnin:RS, ]), lambdat), 3)   #スイッチング変数の混合率の事後平均
round(cbind(apply(THETA1[, , burnin:RS], c(1, 2), mean), thetat1), 3)   #ユーザーのトピック割当の事後平均
round(cbind(apply(THETA2[, , burnin:RS], c(1, 2), mean), thetat2), 3)   #アイテムのトピック割当の事後平均
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phit)), 3)   #ユーザーの単語分布の事後平均
round(cbind(t(apply(GAMMA[, , burnin:RS], c(1, 2), mean)), t(gammat)), 3)   #アイテムの単語分布の事後平均


#サンプリング結果の事後信用区間
round(apply(LAMBDA[burnin:RS, ], 2, function(x) quantile(x, 0.025)), 3)
round(apply(LAMBDA[burnin:RS, ], 2, function(x) quantile(x, 0.975)), 3)
round(apply(THETA1[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(THETA2[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)
round(t(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)


##サンプリングされた潜在変数の要約
n <- max(SEG1)
round(cbind(SEG1/n, Z1), 3)
round(cbind(rowSums(SEG21), SEG21/max(rowSums(SEG21)), Z21 %*% 1:k1), 3)
round(cbind(rowSums(SEG22), SEG22/max(rowSums(SEG22)), Z22 %*% 1:k2), 3)

