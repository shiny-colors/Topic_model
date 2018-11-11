#####Switch introduced Context aware Topic Model#####
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

####データの発生####
##データの設定
L <- 2
k1 <- 15   #同行者トピック数
k2 <- 12   #同行者単語トピック数
k3 <- 15   #単語トピック数
hh <- 5000   #ユーザー数
doc <- rtpois(hh, rgamma(hh, 17.5, 2.0), a=1, b=Inf)   #文書数
d <- sum(doc)   #総文書数
g <- rtpois(d, 1.75, a=0, b=Inf)   #同行者数
m <- 50   #同行者種類数
v1 <- 500   #同行者依存の語彙数
v2 <- 700   #トピック依存の語彙数
v <- v1 + v2   #総語彙数
index_v1 <- 1:v1   #同行者依存の語彙のインデックス
index_v2 <- (max(index_v1)+1):v   #トピック依存の語彙のインデックス
w <- rtpois(d, rgamma(d, 30, 0.3), a=50, b=Inf)   #文書あたりの単語数
f <- sum(w)   #総単語数
vec_k1 <- rep(1, k1)
vec_k2 <- rep(1, k2)
vec_k3 <- rep(1, k3)

##IDを設定
#文書IDを設定
g_id <- rep(1:d, g)
user_id <- rep(1:hh, doc)
no_id <- as.numeric(unlist(tapply(1:d, user_id, rank)))
user_w <- as.numeric(tapply(w, user_id, sum))

#単語IDを設定
u_id <- rep(user_id, rep(1, d) * w)
d_id <- rep(1:d, w)
a_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))

##パラメータの設定
#事前分布のパラメータを設定
alpha1 <- rep(5.0, k1)
alpha2 <- rep(0.15, k2)
alpha3 <- rep(0.20, k3)
beta1 <- rep(0.02, m)
beta2 <- c(rep(0.05, v1), rep(0.001, v2))
beta3 <- c(rep(0.001, v1), rep(0.05, v2))
tau <- c(20.0, 45.0)

##モデルに基づきデータを生成
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  ##パラメータを生成
  #ディレクリ分布からパラメータを生成
  theta1 <- thetat1 <- as.numeric(extraDistr::rdirichlet(1, alpha1))
  theta2 <- thetat2 <- extraDistr::rdirichlet(k1, alpha2)
  theta3 <- thetat3 <- extraDistr::rdirichlet(hh, alpha3)
  gamma <- gammat <- extraDistr::rdirichlet(k1, beta1)
  phi1 <- extraDistr::rdirichlet(k2, beta2)
  phi2 <- extraDistr::rdirichlet(k3, beta3)

  #単語出現確率が低いトピックを入れ替える
  index_v1 <- which(colMaxs(phi1) < (k3*k3*2)/f)[which(colMaxs(phi1) < (k3*k3*3)/f) %in% index_v1]
  index_v2 <- which(colMaxs(phi2) < (k3*k3*2)/f)[which(colMaxs(phi2) < (k3*k3*3)/f) %in% index_v2]
  for(j in 1:length(index_v1)){
    phi1[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(2.0, k2))) %*% 1:k2), index_v1[j]] <- (k3*k3*3)/f
  }
  for(j in 1:length(index_v2)){
    phi2[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(2.0, k3))) %*% 1:k3), index_v2[j]] <- (k3*k3)/f
  }
  phit1 <- phi1; phit2 <- phi2
  
  #スイッチングパラメータを生成
  pi <- pit <- rbeta(hh, tau[1], tau[2])
  
  ##文書ごとにトピックと単語を生成
  Z1_list <- Z2_list <- list()
  word_list <- list()
  switch_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  #同行者トピックと同行者を生成
  M <- extraDistr::rmnom(d, 1, theta1)
  m_vec <- as.numeric(M %*% 1:k1)
  context <- rmnom(d, 1, gamma[m_vec, ])
  context_vec <- as.numeric(context %*% 1:m)

  #単語単位でトピックと単語を生成
  for(i in 1:d){
    
    #スイッチング変数を生成
    s <- rbinom(w[i], 1, pi[user_id[i]])
    index_s <- which(s==1)
    n <- length(index_s)
    
    #同行者トピックを生成
    z1 <- rmnom(w[i], 1, theta2[m_vec[i], ])
    z1_vec <- as.numeric(z1 %*% 1:k2)
    
    #トピックを生成
    z2 <- rmnom(w[i], 1, theta3[user_id[i], ])
    z2_vec <- as.numeric(z2 %*% 1:k3)
    
    #スイッチング変数とトピックから単語を生成
    word <- matrix(0, nrow=w[i], ncol=v)
    word[index_s, ] <- rmnom(n, 1, phi1[z1_vec[index_s], ])
    word[-index_s, ] <- rmnom(w[i]-n, 1, phi2[z2_vec[-index_s], ])
    
    #データを格納
    word_list[[i]] <- as.numeric(word %*% 1:v)
    WX[i, ] <- colSums2(word)
    switch_list[[i]] <- s
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
  }
  
  #break条件
  if(min(colSums2(WX)) > 0){
    break
  }
}

#データを変換
S <- unlist(switch_list)
Z1 <- do.call(rbind, Z1_list); storage.mode(Z1) <- "integer"
Z2 <- do.call(rbind, Z2_list); storage.mode(Z2) <- "integer"
storage.mode(WX) <- "integer"
wd <- unlist(word_list)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
word_data_T <- t(word_data)
rm(word_list); rm(Z1_list); rm(Z2_list)
gc(); gc()


####マルコフ連鎖モンテカルロ法でSwitch introduced Context aware Topic Modelを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / as.numeric(Bur %*% vec_k)   #負担率
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}

##アルゴリズムの設定
R <- 2000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##インデックスの設定
d_dt <- sparseMatrix(d_id, 1:f, x=rep(1, f), dims=c(d, f))
u_dt <- sparseMatrix(u_id, 1:f, x=rep(1, f), dims=c(hh, f))
w_dt <- t(word_data)

##事前分布の設定
#ディリクレ分布の事前分布
alpha01 <- 0.1
beta01 <- 0.01
s0 <- 0.1
v0 <- 0.1
er <- 0.00001

##パラメータの真値
#トピック分布の真値
theta1 <- thetat1
theta2 <- thetat2
theta3 <- thetat3
pi <- pit
pi_vec <- pi[u_id]

#単語分布の真値
phi1 <- phit1
phi2 <- phit2
gamma <- gammat

#潜在変数の真値
Zi1 <- Z1 * S
z_vec1 <- as.numeric(Zi1 %*% 1:k2)
index_y <- which(S==1)

##パラメータの初期値
#トピック分布の初期値
theta1 <- as.numeric(extraDistr::rdirichlet(1, rep(10.0, k1)))
theta2 <- extraDistr::rdirichlet(k1, rep(2.5, k2))
theta3 <- extraDistr::rdirichlet(u, rep(2.5, k3))
pi <- rep(0.5, hh)
pi_vec <- pi[u_id]

#単語分布の初期値
phi1 <- extraDistr::rdirichlet(k2, rep(10.0, v))
phi2 <- extraDistr::rdirichlet(k3, rep(10.0, v))
gamma <- extraDistr::rdirichlet(k1, rep(2.5, m))

#潜在変数の初期値
y <- rbinom(f, 1, 0.5)
Zi1 <- rmnom(f, 1, rep(k2/1, k2)) * y
z_vec1 <- as.numeric(Zi1 %*% 1:k2)
index_y <- which(y==1)


##対数尤度の基準値
#ユニグラムモデルの対数尤度
LLst <- sum(word_data %*% log(colSums2(WX)/f))

#ベストモデルの対数尤度
LLbest1 <- sum(log(as.numeric((thetat2[as.numeric(M %*% 1:k1)[d_id], ] * t(phit1)[wd, ])[S==1, ] %*% vec_k2)))
LLbest2 <- sum(log(as.numeric((thetat3[u_id, ] * t(phit2)[wd, ])[S==0, ] %*% vec_k3)))
LLbest <- LLbest1 + LLbest2


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##同行者トピックを生成
  #潜在変数の割当確率を設定
  Lho_z1 <- exp(as.matrix(d_dt %*% (Zi1 %*% log(t(theta2)))))
  Lho_context <- matrix(theta1, nrow=d, ncol=k1, byrow=T) * t(gamma)[context_vec, ] * Lho_z1
  context_rate <- Lho_context / as.numeric(Lho_context %*% vec_k1)

  #多項分布からトピックを生成
  Mi <- rmnom(d, 1, context_rate)
  m_vec <- as.numeric(Mi %*% 1:k1)[d_id]   #単語長の同行者トピックベクトル
  m_dt <- sparseMatrix(m_vec, 1:f, x=rep(1, f), dims=c(k1, f)) 
  
  ##スイッチング変数を生成
  #トピックごとの尤度を設定
  Lho1 <- theta2[m_vec, ] * t(phi1)[wd, ]
  Lho2 <- theta3[u_id, ] * t(phi2)[wd, ]

  #潜在変数の割当確率を設定
  Lho_mu1 <- as.numeric(Lho1 %*% vec_k2)
  Lho_mu2 <- as.numeric(Lho2 %*% vec_k3)
  switching_rate <- (pi_vec*Lho_mu1) / (pi_vec*Lho_mu1 + (1-pi_vec)*Lho_mu2)
  
  #ベルヌーイ分布からスイッチング変数を生成
  y <- rbinom(f, 1, switching_rate)
  index_y <- which(y==1)
  sn <- length(index_y)
  
  
  #ベータ分布から混合率をサンプリング
  n <- as.numeric(u_dt %*% y)
  tau1 <- n + s0
  tau2 <- user_w - n + v0
  pi <- rbeta(hh, tau1, tau2)   #パラメータをサンプリング
  pi_vec <- pi[u_id]
  
  ##同行者依存単語および一般語のトピックを生成
  #潜在変数の割当確率を設定
  Lho_rate1 <- Lho1[index_y, ] / Lho_mu1[index_y]
  Lho_rate2 <- Lho2[-index_y, ] / Lho_mu2[-index_y]
  
  #多項分布からトピックを生成
  Zi1 <- matrix(0, nrow=f, ncol=k2)
  Zi2 <- matrix(0, nrow=f, ncol=k3)
  Zi1[index_y, ] <- rmnom(sn, 1, Lho_rate1)
  Zi2[-index_y, ] <- rmnom(f-sn, 1, Lho_rate2)
  
  
  ##トピック分布のパラメータを生成
  #ディリクレ分布のパラメータを設定
  wsum1 <- colSums2(Mi) + alpha01
  wsum2 <- as.matrix(m_dt %*% Zi1) + alpha01
  wsum3 <- as.matrix(u_dt %*% Zi2) + alpha01
  
  #ディリクレ分布からトピック分布を生成
  theta1 <- as.numeric(extraDistr::rdirichlet(1, wsum1))
  theta2 <- extraDistr::rdirichlet(k1, wsum2)
  theta3 <- extraDistr::rdirichlet(hh, wsum3)
  
  
  ##同行者分布および単語分布のパラメータを生成
  #ディリクレ分布のパラメータ
  vsum1 <- t(Mi) %*% context + beta01
  vsum2 <- as.matrix(t(w_dt %*% Zi1)) + beta01
  vsum3 <- as.matrix(t(w_dt %*% Zi2)) + beta01
  
  #ディリクレ分布から同行者分布および単語分布を生成
  gamma <- extraDistr::rdirichlet(k1, vsum1)
  phi1 <- extraDistr::rdirichlet(k2, vsum2)
  phi2 <- extraDistr::rdirichlet(k3, vsum3)

  ##パラメータの格納とサンプリング結果の表示
  if(rp%%disp==0){
    #対数尤度を計算
    LL1 <- sum(log(as.numeric((theta2[m_vec, ] * t(phi1)[wd, ])[index_y, ] %*% vec_k2)))
    LL2 <- sum(log(as.numeric((theta3[u_id, ] * t(phi2)[wd, ])[-index_y, ] %*% vec_k3)))
    LL <- LL1 + LL2
    
    #サンプリング結果を確認
    print(rp)
    print(c(mean(y), mean(S)))
    print(round(phi2[, (v1-9):(v1+10)], 3))
    print(c(LL, LLbest, LLst))
  }
}



