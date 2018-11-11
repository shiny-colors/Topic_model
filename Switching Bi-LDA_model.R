#####Switching Bi-LDAモデル#####
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(bayesm)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)

#set.seed(21437)

####データの発生####
##データの設定
hh0 <- 1000
item0 <- 75

##IDとレビュー履歴を発生
#IDを仮設定
u.id0 <- rep(1:hh0, rep(item0, hh0))
i.id0 <- rep(1:item0, hh0)

#レビュー履歴を発生
buy_hist <- rep(0, hh0*item0)
for(i in 1:item0){
  p <- runif(1, 0.05, 0.3)
  buy_hist[i.id0==i] <- rbinom(hh0, 1, p)
}

#レビュー履歴からIDを再設定
index <- which(buy_hist==1)
u.id <- u.id0[index]
u.freq <- plyr::count(u.id)[, 2]
i.id <- i.id0[index]
i.freq <- plyr::count(i.id)[, 2]
ID <- data.frame(no=1:length(u.id), id=u.id, item=i.id)

#データの再設定
k1 <- 8   #ユーザートピック数
k2 <- 10   #アイテムトピック数
hh <- length(unique(u.id))   #ユーザー数
item <- length(unique(i.id))   #アイテム数
d <- length(u.id)   #文書数
v <- 300   #語彙数
v1 <- 175   #ユーザー評価に関係のある語彙数
v2 <- v-v1   #アイテム評価に関係のある語彙数

#1文書あたりの単語数
w <- rpois(d, rgamma(d, 50, 1.0))   #1文書あたりの単語数
w[w < 25] <- ceiling(runif(sum(w < 25), 25, 35))
f <- sum(w)   #総単語数


####bag of word形式の文書行列を発生####
##パラメータの設定
#ディリクレ事前分布のパラメータを設定
alpha01 <- rep(0.25, k1)   #ユーザートピックのディリクレ事前分布のパラメータ
alpha02 <- rep(0.3, k2)   #アイテムトピックのディリクレ事前分布のパラメータ
alpha11 <- c(rep(0.4, v1), rep(0.0075, v2))   #ユーザーの単語分布のディリクレ事前分布のパラメータ
alpha12 <- c(rep(0.0075, v1), rep(0.45, v2))   #アイテムの単語分布のディリクレ事前分布のパラメータ

#ディリクレ事前分布からパラメータを生成
thetat <- theta <- extraDistr::rdirichlet(hh, alpha01)   #ユーザートピックの生成
gammat <- gamma <- extraDistr::rdirichlet(item, alpha02)   #アイテムトピックの生成
phit <- phi <- extraDistr::rdirichlet(k1, alpha11)   #ユーザーの単語分布を生成
lambdat <- lambda <- extraDistr::rdirichlet(k2, alpha12)   #アイテムの単語分布を生成

#ユーザーかアイテムのスイッチング変数のパラメータ
omega <- rbeta(hh, 8, 10)


##多項分布からトピックおよび単語データを生成
WX <- matrix(0, nrow=d, ncol=v)
Z01_list <- list()
Z02_list <- list()
y0_list <- list()
u.id_list <- list()
i.id_list <- list()
index_word1 <- 1:v1
index_word2 <- (v1+1):v

##レビューごとに単語を生成する
for(i in 1:hh){
  print(i)
  
  for(j in 1:u.freq[i]){
    word <- w[u.id==i][j]   #生成する単語数
    index_row <- which(u.id==i)[j]
    
    #ユーザートピックを生成
    z1 <- rmnom(word, 1, theta[i, ])
    z1_vec <- z1 %*% 1:k1
    
    #アイテムトピックを生成
    z2 <- rmnom(word, 1, gamma[i.id[index_row], ])
    z2_vec <- z2 %*% 1:k2
    
    #スイッチング変数を生成
    y <- rbinom(word, 1, omega[i])
    index_y <- which(y==1)
    
    #パラメータから単語を生成
    user_word <- colSums(rmnom(length(index_y), 1, phi[z1_vec[index_y], ]))
    item_word <- colSums(rmnom(word-length(index_y), 1, lambda[z2_vec[-index_y], ]))
    
    #データを格納
    WX[index_row, ] <- user_word + item_word
    Z01_list[[index_row]] <- z1
    Z02_list[[index_row]] <- z2
    y0_list[[index_row]] <- y
    u.id_list[[index_row]] <- rep(i, word)
    i.id_list[[index_row]] <- rep(i.id[index_row], word)
  }
}

##リスト形式をデータ形式に変換
Z02 <- do.call(rbind, Z02_list)
Z01 <- do.call(rbind, Z01_list)
y0 <- unlist(y0_list)
u.id_vec <- unlist(u.id_list)
i.id_vec <- unlist(i.id_list)
storage.mode(Z02) <- "integer"
storage.mode(Z01) <- "integer"
storage.mode(WX) <- "integer"


####トピックモデル推定のためのデータの準備####
##データ推定用のIDを作成
user_list <- list()
item_list <- list()
wd_list <- list()

#IDごとにtweet_idおよび単語idを作成
for(i in 1:hh){
  print(i)
  
  for(j in 1:u.freq[i]){
    index_row <- which(u.id==i)[j]
    
    #ユーザーIDとアイテムIDを記録
    user_list[[index_row]] <- rep(i, w[index_row])
    item_list[[index_row]] <- rep(i.id[index_row], w[index_row])
    
    #単語IDを記録
    num1 <- WX[index_row, ] * 1:v
    num2 <- which(num1 > 0)
    W1 <- WX[index_row, (WX[index_row, ] > 0)]
    wd_list[[index_row]] <- rep(num2, W1)
  }
}

#リストをベクトルに変換
user_id <- unlist(user_list)
item_id <- unlist(item_list)
wd <- unlist(wd_list)

##インデックスを作成
user_list <- list()
item_list <- list()
word_list <- list()
for(i in 1:length(unique(user_id))){user_list[[i]] <- which(user_id==i)}
for(i in 1:length(unique(item_id))){item_list[[i]] <- which(item_id==i)}
for(i in 1:length(unique(wd))){word_list[[i]] <- which(wd==i)}


####マルコフ連鎖モンテカルロ法でSwitching Bi-LDAモデルを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #負担係数の格納用
  for(j in 1:k){
    #負担係数を計算
    Bi <- rep(theta[, j], w) * phi[j, wd]   #尤度
    Bur[, j] <- Bi   
  }
  
  Br <- Bur / rowSums(Bur)   #負担率の計算
  r <- colSums(Br) / sum(Br)   #混合率の計算
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##アルゴリズムの設定
R <- 10000
keep <- 2
iter <- 0
burnin <- 1000/keep

##事前分布の設定
alpha01 <- 1
alpha02 <- 1
beta01 <- 0.5
gamma01 <- 0.5

##パラメータの初期値
#トピック分布の初期値
theta <- extraDistr::rdirichlet(hh, rep(5, k1))   #ユーザートピックの初期値
gamma <- extraDistr::rdirichlet(item, rep(5, k2))   #アイテムトピックの初期値

#ユーザーの単語分布の初期値
u_word <- as.matrix(data.frame(id=u.id, WX) %>%
                      dplyr::group_by(id) %>%
                      dplyr::summarise_all(funs(sum)))[, 2:(v+1)]
phi <- extraDistr::rdirichlet(k1, apply(u_word, 2, sd))

#アイテムの単語分布の初期値
i_word <- as.matrix(data.frame(id=i.id, WX) %>%
                      dplyr::group_by(id) %>%
                      dplyr::summarise_all(funs(sum)))[, 2:(v+1)]
lambda <- extraDistr::rdirichlet(k2, apply(i_word, 2, sd)/10)

#ユーザーとアイテムの混合率の初期値
y <- rbinom(f, 1, 0.5)
r <- matrix(c(0.5, 0.5), nrow=hh, ncol=2)


##パラメータの保存用配列
THETA <- array(0, dim=c(hh, k1, R/keep))
GAMMA <- array(0, dim=c(item, k2, R/keep))
PHI <- array(0, dim=c(k1, v, R/keep))
LAMBDA <- array(0, dim=c(k2, v, R/keep))
SEG1 <- matrix(0, nrow=f, ncol=k1)
SEG2 <- matrix(0, nrow=f, ncol=k2)
Y <- rep(0, f)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"

##MCMC推定用配列
phi_d <- matrix(0, nrow=f, ncol=k1)
lambda_d <- matrix(0, nrow=f, ncol=k2)
Bur1 <- matrix(0, nrow=f, ncol=k1)
Bur2 <- matrix(0, nrow=f, ncol=k2)
LH1 <- rep(0, f)
usum0 <- matrix(0, nrow=hh, ncol=k1)
isum0 <- matrix(0, nrow=item, ncol=k2)
uvf0 <- matrix(0, nrow=k1, ncol=v)
ivf0 <- matrix(0, nrow=k2, ncol=v)
vec1 <- 1/1:k1
vec2 <- 1/1:k2


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとにユーザートピックとアイテムトピックを生成
  #レビュー単位に拡大したパラメータ行列
  theta_d <- theta[u.id, ]
  gamma_d <- gamma[i.id, ]
  
  #ユーザー単位の尤度とトピック割当確率を推定
  for(j in 1:k1){
    phi_d[, j] <- phi[j, wd]
    Bi1 <- rep(theta_d[, j], w) * phi_d[, j]
    Bur1[, j] <- Bi1
  }
  user_rate <- Bur1 / rowSums(Bur1)
  
  #アイテム単位の尤度とトピック割当確率を推定
  for(j in 1:k2){
    lambda_d[, j] <- lambda[j, wd]
    Bi2 <- rep(gamma_d[, j], w) * lambda_d[, j]
    Bur2[, j] <- Bi2
  }
  item_rate <- Bur2 / rowSums(Bur2)
  
  ##スイッチング変数を生成
  #スイッチング変数のパラメータを設定
  rd <- r[user_id, ]
  #LLz1 <- rd[, 1] * rowSums(user_rate*phi_d)
  #LLz2 <- rd[, 2] * rowSums(item_rate*lambda_d)
  LLz1 <- rd[, 1] * rowSums(Bur1)
  LLz2 <- rd[, 2] * rowSums(Bur2)
  switching_rate <- LLz1 / (LLz1+LLz2)
  
  #ベルヌーイ分布よりスイッチング変数を生成
  y <- rbinom(f, 1, switching_rate)
  r1 <- tapply(y, user_id, mean)
  r <- cbind(r1, 1-r1) 
  
  ##生成したスイッチング変数に基づきトピックを生成
  index_y <- which(y==1)
  Zi1 <- matrix(0, nrow=f, ncol=k1)
  Zi2 <- matrix(0, nrow=f, ncol=k2)
  z1_vec <- rep(0, f)
  z2_vec <- rep(0, f)
  
  #多項分布よりユーザートピックを生成
  n <- length(index_y)
  rand1 <- matrix(runif(n), nrow=n, ncol=k1)
  user_cumsums <- rowCumsums(user_rate[index_y, ])
  z1 <- ((k1+1) - (user_cumsums > rand1) %*% rep(1, k1)) %*% vec1   #トピックをサンプリング
  z1[z1!=1] <- 0
  Zi1[index_y, ] <- z1
  z1_vec[index_y] <- z1 %*% 1:k1
  
  #多項分布よりアイテムトピックを生成
  n <- f-length(index_y)
  rand2 <- matrix(runif(n), nrow=n, ncol=k2)
  item_cumsums <- rowCumsums(item_rate[-index_y, ])
  z2 <- ((k2+1) - (item_cumsums > rand2) %*% rep(1, k2)) %*% vec2   #トピックをサンプリング
  z2[z2!=1] <- 0
  Zi2[-index_y, ] <- z2
  z2_vec[-index_y] <- z2 %*% 1:k2
  
  
  ##ユーザートピックのパラメータをサンプリング
  for(i in 1:hh){
    usum0[i, ] <- colSums(Zi1[user_list[[i]], ])
  }
  usum <- usum0 + alpha01
  theta <- extraDistr::rdirichlet(hh, usum)
  
  ##アイテムトピックのパラメータをサンプリング
  for(i in 1:item){
    isum0[i, ] <- colSums(Zi2[item_list[[i]], ])
  }
  isum <- isum0 + alpha02
  gamma <- extraDistr::rdirichlet(item, isum)
  
  
  ##ユーザーとアイテムレベルの単語分布をサンプリング
  for(j in 1:v){
    uvf0[, j] <- colSums(Zi1[word_list[[j]], ])
    ivf0[, j] <- colSums(Zi2[word_list[[j]], ])
  }
  uvf <- uvf0 + beta01
  ivf <- ivf0 + gamma01
  phi <- extraDistr::rdirichlet(k1, uvf)
  lambda <- extraDistr::rdirichlet(k2, ivf)
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    GAMMA[, , mkeep] <- gamma
    PHI[, , mkeep] <- phi
    LAMBDA[, , mkeep] <- lambda
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      Y <- Y + y
    }
    
    #サンプリング結果を確認
    print(rp)
    print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
    #print(round(cbind(gamma[1:10, ], gammat[1:10, ]), 3))
    print(round(cbind(phi[, 171:180], phit[, 171:180]), 3))
    #print(round(cbind(lambda[, 171:180], lambdat[, 171:180]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep
RS <- R/keep

##サンプリング結果のプロット
matplot(t(THETA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[250, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[25, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[50, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 1, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 175, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[, 176, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(LAMBDA[, 175, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(LAMBDA[, 176, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(LAMBDA[, 200, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

##サンプリング結果の要約
#サンプリング結果の事後平均
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)   #ユーザーのトピック割当の事後平均
round(cbind(apply(GAMMA[, , burnin:RS], c(1, 2), mean), gammat), 3)   #アイテムのトピック割当の事後平均
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phit)), 3)   #ユーザーの単語分布の事後平均
round(cbind(t(apply(LAMBDA[, , burnin:RS], c(1, 2), mean)), t(lambdat)), 3)   #アイテムの単語分布の事後平均

#サンプリング結果の事後信用区間
round(apply(THETA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(THETA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)
round(t(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)

##サンプリングされた潜在変数の要約
n <- length(burnin:RS)
round(cbind(Y / n, wd), 3)
round(cbind(wd, y=Y/n, SEG1/rowSums(SEG1)), 3)
round(cbind(wd, y=Y/n, SEG2/rowSums(SEG2)), 3)

