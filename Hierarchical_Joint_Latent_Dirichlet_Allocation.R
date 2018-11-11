#####Hierarchical Joint Latent Dirichlet Allocation#####
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
k1 <- 20   #ユーザー単位のトピック数
k2 <- 30   #セッション単位のトピック数
k3 <- 30   #アイテム単位のトピック数
hh <- 3000   #ユーザー数
item <- 1500   #アイテム数
pages <- 25   #ページ数
pt <- rtpois(hh, rgamma(hh, 20, 0.25), a=1, b=Inf)   #セッション数
hhpt <- sum(pt)   #総セッション数
w <- extraDistr::rtpois(hhpt, rgamma(hhpt, 2.75, 0.5), a=0, b=Inf)   #セッションあたりの閲覧アイテム数
f <- sum(w)   #総閲覧アイテム数

##IDとインデックスの設定
#セッション単位のIDを設定
user_id <- rep(1:hh, pt)

#アイテム単位のIDの設定
user_id_vec <- rep(rep(1:hh, pt), w)
session_id <- rep(1:hhpt, w)
user_no <- as.numeric(unlist(tapply(1:f, user_id_vec, rank)))
session_no <- as.numeric(unlist(tapply(1:f, session_id, rank)))

#インデックスの設定
user_index <- session_index <- list()
for(i in 1:hh){
  user_index[[i]] <- which(user_id_vec==i)
}
vec <- c(0, cumsum(w))
for(i in 1:hhpt){
  session_index[[i]] <- (1:f)[(vec[i]+1):vec[i+1]]
}

##パラメータを生成
#ディリクレ分布のパラメータを設定
alpha01 <- rep(0.1, k1)
alpha02 <- rep(0.1, k3)
beta01 <- rep(0.1, k2)
beta02 <- rep(0.2, pages)
beta03 <- matrix(0.015, nrow=k3, ncol=item, byrow=T)
for(j in 1:k2){
  beta03[j, matrix(1:item, nrow=k3, ncol=item/k3, byrow=T)[j, ]] <- 0.1
}


##すべてのアイテムが出現するまでデータの生成を続ける
rp <- 0
repeat {
  rp <- rp + 1
  
  #ディリクレ分布からパラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(hh, alpha01)
  theta2 <- thetat2 <- extraDistr::rdirichlet(k2, alpha02)
  gamma <- gammat <- extraDistr::rdirichlet(k1, beta01)
  omega <- omegat <- extraDistr::rdirichlet(k1, beta02)
  phi <- phit <- extraDistr::rdirichlet(k3, beta03)

  #アイテム出現確率が低いトピックを入れ替える
  index <- which(colMaxs(phi) < (k2*7)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, beta01)) %*% 1:k2), index[j]] <- (k2*7)/f
  }
  
  ##応答変数を生成
  y_list <- v_list <- p_list <- z_list <- w_list <- list()
  w_sums <- rep(0, item)
  
  for(i in 1:hh){
    #インデックスを抽出
    u_index <- user_index[[i]]
    
    #セッションのトピックを生成
    y <- rmnom(pt[i], 1, theta1[i, ])
    y_vec <- as.numeric(y %*% 1:k1)
    
    #セッショントピックからセッションを生成
    v <- rmnom(pt[i], 1, gamma[y_vec, ])
    v_vec <- as.numeric(v %*% 1:k2)
    
    #セッションからアイテムトピックとページを生成
    s_index <- session_id[u_index] - (min(session_id[u_index])-1); n <- length(s_index)
    p <- rmnom(n, 1, omega[y_vec[s_index], ])
    p_vec <- as.numeric(p %*% 1:pages)
    z <- rmnom(n, 1, theta2[v_vec[s_index], ])
    z_vec <- as.numeric(z %*% 1:k2)
    
    #アイテムトピックからアイテムを生成
    w <- rmnom(n, 1, phi[z_vec, ])
    w_vec <- as.numeric(w %*% 1:item)
  
    #生成したデータを格納
    y_list[[i]] <- y
    v_list[[i]] <- v
    p_list[[i]] <- p_vec
    z_list[[i]] <- z_vec
    w_list[[i]] <- w_vec
    w_sums <- w_sums + colSums(w)
  }
  #break条件
  print(c(rp, sum(w_sums==0)))
  if(sum(w_sums==0)==0){
    break
  }
}

#リストを変換
y <- do.call(rbind, y_list); y_vec <- (y %*% 1:k1)
v <- do.call(rbind, v_list); v_vec <- (v %*% 1:k2)[session_id]
z_vec <- unlist(z_list)
pd <- unlist(p_list)
wd <- unlist(w_list)
page_data <- sparseMatrix(1:f, pd, x=rep(1, f), dims=c(f, pages))
page_data_T <- t(page_data)
item_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, item))
item_data_T <- t(item_data)
hist(w_sums, col="grey", breaks=50, xlab="アイテム出現頻度", main="アイテム出現分布")


####ギブスサンプリングでHierarchical Joint Latent Dirichlet Allocationを推定####
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
R <- 2000
keep <- 2  
iter <- 0
burnin <- 300
disp <- 10

##事前分布の設定
#ディリクレ分布の事前分布
alpha01 <- 0.1
beta01 <- 0.1
er <- 0.0001

##パラメータの真値
theta1 <- thetat1
theta2 <- thetat2
gamma <- gammat
omega <- omegat
phi <- phit

##パラメータの初期値を設定
#トピック分布の初期値
theta1 <- extraDistr::rdirichlet(hh, rep(2.0, k1))
theta2 <- extraDistr::rdirichlet(k2, rep(2.0, k3))

#アイテム分布の初期値
gamma <- extraDistr::rdirichlet(k1, rep(2.0, k2))
omega <- extraDistr::rdirichlet(k1, rep(2.0, pages))
phi <- extraDistr::rdirichlet(k3, rep(2.0, item))

#トピックの初期値
y_vec <- as.numeric(rmnom(hhpt, 1, rep(1, k1)) %*% 1:k1)


##パラメータの格納用配列
THETA1 <- array(0, dim=c(hh, k1, R/keep))
THETA2 <- array(0, dim=c(k2, k3, R/keep))
GAMMA <- array(0, dim=c(k1, k2, R/keep))
OMEGA <- array(0, dim=c(k1, pages, R/keep))
PHI <- array(0, dim=c(k3, item, R/keep))
SEG1 <- matrix(0, nrow=hhpt, ncol=k1)
SEG2 <- matrix(0, nrow=hhpt, ncol=k2)
SEG3 <- matrix(0, nrow=f, ncol=k3)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"
storage.mode(SEG3) <- "integer"

##データとインデックスの設定
#インデックスの設定
user_dt <- sparseMatrix(user_id, 1:hhpt, x=rep(1, hhpt), dims=c(hh, hhpt))
session_dt <- sparseMatrix(session_id, 1:f, x=rep(1, f), dims=c(hhpt, f))


##対数尤度の基準値
LLst <- sum(item_data %*% log(colMeans(item_data)))
LLbest <- sum(log(as.numeric((thetat2[as.numeric(v %*% 1:k2)[session_id], ] * t(phit)[wd, ]) %*% rep(1, k3))))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##セッションの潜在変数を生成
  #データの設定
  theta1_vec <- theta1[user_id, ]   #アイテムトピック
  phi_vec <- t(phi)[wd, ]   #単語分布　
  
  #セッションの割当確率を生成
  Lho_topic <- exp(session_dt %*% log(phi_vec %*% t(theta2)))
  Lho_session <- gamma[y_vec, ] * Lho_topic
  session_rate <- Lho_session / as.numeric(Lho_session %*% rep(1, k2))
  session_rate <- (session_rate + er) / (session_rate + er) %*% rep(1, k2)
  
  #多項分布からセッションを生成
  V <- extraDistr::rmnom(hhpt, 1, session_rate)
  v_vec <- as.numeric(V %*% 1:k2)
  
  ##アイテムトピックを生成
  #アイテムトピックの割当確率
  v_session <- v_vec[session_id]
  Lho_w <- theta2[v_session, ] * phi_vec   #アイテムトピックの期待尤度
  topic_rate <- Lho_w / as.numeric(Lho_w %*% rep(1, k2))
  
  #多項分布からアイテムトピックを生成
  Z <- extraDistr::rmnom(f, 1, topic_rate)
  z_vec <- as.numeric(Z %*% 1:k2)
  
  ##セッショントピックを生成
  #セッショントピックの割当確率
  Lho_s <- theta1_vec * exp(session_dt %*% t(log(omega))[pd, ]) * t(gamma)[v_vec, ]   #セッショントピックの期待尤度
  topic_rate <- Lho_s / as.numeric(Lho_s %*% rep(1, k1))
  topic_rate <- (topic_rate + er) / (topic_rate + er) %*% rep(1, k1)
  
  #多項分布からセッショントピックを生成
  Y <- extraDistr::rmnom(hhpt, 1, topic_rate)
  y_vec <- as.numeric(Y %*% 1:k1)
  
  ##トピック分布のパラメータをサンプリング
  #ディリクレ分布のパラメータを設定
  v_dt <- sparseMatrix(v_session, 1:f, x=rep(1, f), dims=c(k2, f)) 
  y_sums <- user_dt %*% Y + alpha01
  z_sums <- v_dt %*% Z + alpha01
  
  #ディリクレ分布からトピック分布を生成
  theta1 <- extraDistr::rdirichlet(hh, y_sums)
  theta2 <- extraDistr::rdirichlet(k2, z_sums)
  
  
  ##出現確率分布のパラメータをサンプリング
  #ディリクレ分布のパラメータを設定
  y_dt <- sparseMatrix(1:f, y_vec[session_id], x=rep(1, f), dims=c(f, k1))
  p_sums <- t(page_data_T %*% y_dt) + beta01
  s_sums <- t(v_dt %*% y_dt) + beta01
  w_sums <- t(item_data_T %*% Z) + beta01
    
  #ディリクレ分布から出現確率分布をサンプリング
  omega <- extraDistr::rdirichlet(k1, p_sums)
  gamma <- extraDistr::rdirichlet(k1, s_sums)
  phi <- extraDistr::rdirichlet(k3, w_sums)

  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma
    OMEGA[, , mkeep] <- omega
  }  
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG1 <- SEG1 + Y
    SEG2 <- SEG2 + V
    SEG3 <- SEG3 + Z
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL <- sum(log(as.numeric((theta2[v_vec[session_id], ] * t(phi)[wd, ]) %*% rep(1, k3))))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL, LLbest, LLst))
  }
}

