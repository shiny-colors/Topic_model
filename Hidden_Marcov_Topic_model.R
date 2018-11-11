#####Hidden Marcov Topic Model#####
options(warn=2)
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
k1 <- 15   #HMMの混合数
k2 <- 15   #共通のトピック数
d <- 2000   #文書数
v <- 500   #語彙数
s <- rpois(d, 15)   #文章数
s[s < 5] <- ceiling(runif(sum(s < 5), 5, 10))
a <- sum(s)   #総文章数
w <- rpois(a, 12)   #文章あたりの単語数
w[w < 5] <- ceiling(runif(sum(w < 5), 5, 10))
f <- sum(w)   #総単語数

#文書IDの設定
u_id <- rep(1:d, s)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:s[i])}
words <- as.numeric(tapply(w, u_id, sum))

#文章区切りのベクトルを作成
ID_d <- rep(1:d, words)
td_d <- c()
for(i in 1:d){
  td_d <- c(td_d, rep(1:s[i], w[u_id==i]))
}
nd_d <- rep(1:a, w)
x_vec <- rep(0, f)
x_vec[c(1, cumsum(w[-a])+1)] <- 1

#インデックスを設定
s_list <- list()
vec_list <- list()
for(i in 1:a){
  if(i%%1000==0){
    print(i)
  }
  s_list[[i]] <- which(nd_d==i)
  vec_list[[i]] <- rep(1, length(s_list[[i]]))
}

##パラメータの設定
#ディレクリ分布のパラメータ
alpha01 <- rep(1, k1)
alpha02 <- matrix(0.2, nrow=k1, ncol=k1)
diag(alpha02) <- 2.25
alpha03 <- rep(0.15, k2)
alpha11 <- rep(0.1, v)


for(l in 1:100){
  print(l)
  #パラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(1, alpha01)
  theta2 <- thetat2 <- extraDistr::rdirichlet(k1, alpha02)
  theta3 <- thetat3 <- extraDistr::rdirichlet(k1, alpha03)
  phi0 <- t(extraDistr::rdirichlet(v, rep(0.01, k2))) * 
                          (matrix(extraDistr::rdirichlet(1, rep(2.0, v)), nrow=k2, ncol=v, byrow=T))
  phi <- phit <- phi0 / rowSums(phi0)

  ##モデルにもとづき単語を生成する
  WX <- matrix(0, nrow=a, ncol=v)
  Z1_list <- list()
  Z2_list <- list()
  wd_list <- list()
  
  for(i in 1:a){
    #文章ごとのセグメントを生成
    if(t_id[i]==1){
      z1 <- rmnom(1, 1, theta1)
      Z1_list[[i]] <- as.numeric(z1 %*% 1:k1)
    } else {
      z1 <- rmnom(1, 1, theta2[Z1_list[[i-1]], ])
      Z1_list[[i]] <- as.numeric(z1 %*% 1:k1)
    }
    
    #単語ごとにトピックと単語を生成
    z2 <- rmnom(w[i], 1, theta3[Z1_list[[i]], ])
    Z2_list[[i]] <- as.numeric(z2 %*% 1:k2)
    
    #トピックに基づき単語を生成
    word <- rmnom(w[i], 1, phi[Z2_list[[i]], ])
    WX[i, ] <- colSums(word)
    wd_list[[i]] <- as.numeric(word %*% 1:v)
  }
  if(min(colSums(WX)) > 0) break
}
colSums(WX)


#リストを変換
wd <- unlist(wd_list)
z1 <- unlist(Z1_list)
z2 <- unlist(Z2_list)
Data <- matrix(as.numeric(table(1:f, wd)), nrow=f, ncol=v) 
sparse_data <- as(Data, "CsparseMatrix")
rm(Data)


##インデックスを作成
doc_list <- list()
td_list <- s_list
word_list <- list()
for(i in 1:d){doc_list[[i]] <- which(ID_d==i)}
for(i in 1:v){word_list[[i]] <- which(wd==i)}


####マルコフ連鎖モンテカルロ法でMHMMトピックモデルを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  Bur <-  matrix(0, nrow=length(wd), ncol=k)   #負担係数の格納用
  for(j in 1:k){
    #負担係数を計算
    Bi <- rep(theta[, j], w) * phi[j, wd]   #尤度
    Bur[, j] <- Bi   
  }
  Br <- Bur / rowSums(Bur)   #負担率の計算
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}

##観測データの対数尤度と潜在変数zを計算するための関数
LLobz <- function(Data, phi, r, const, hh, k){
  
  #多項分布の対数尤度
  log_phi <- log(t(phi))
  LLi <- const + Data %*% log_phi
  
  #logsumexpの尤度
  LLi_max <- matrix(apply(LLi, 1, max), nrow=hh, ncol=k)
  r_matrix <- matrix(r, nrow=hh, ncol=k, byrow=T)
  
  #割当確率のパラメータを設定
  expl <- exp(LLi - LLi_max)
  z <- expl / rowSums(expl)   #セグメント割当確率
  
  #観測データの対数尤度
  r_log <- matrix(log(r), nrow=hh, ncol=k, byrow=T)
  LLosum <- sum(log(rowSums(exp(r_log + LLi))))   #観測データの対数尤度
  rval <- list(LLob=LLosum, z=z, LL=LLi)
  return(rval)
}


####HMMトピックモデルのMCMCアルゴリズムの設定####
##アルゴリズムの設定
R <- 10000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##パラメータの真値
theta1 <- thetat1
theta2 <- thetat2
theta3 <- thetat3
phi <- phit
z1_vec <- z1


##MHMTモデルの初期値を設定
##混合多項分布でセグメント割当を初期化
const <- lfactorial(w) - rowSums(lfactorial(WX))   #多項分布の密度関数の対数尤度の定数

#パラメータの初期値
#phiの初期値
alpha0 <- colSums(WX) / sum(WX) + 0.001
phi <- extraDistr::rdirichlet(k1, alpha0*v)

#混合率の初期値
r <- rep(1/k1, k1)

#観測データの対数尤度の初期化
L <- LLobz(WX, phi, r, const, a, k1)
LL1 <- L$LLob
z <- L$z

#更新ステータス
dl <- 100   #EMステップでの対数尤度の差の初期値
tol <- 1
iter <- 0 

##EMアルゴリズムで対数尤度を最大化
while(abs(dl) >= tol){   #dlがtol以上の場合は繰り返す
  #Eステップの計算
  z <- L$z   #潜在変数zの出力
  
  #Mステップの計算と最適化
  #phiの推定
  df0 <- matrix(0, nrow=k1, ncol=v)
  for(j in 1:k1){
    #完全データの対数尤度からphiの推定量を計算
    phi[j, ] <- colSums(matrix(z[, j], nrow=a, ncol=v) * WX) / sum(z[, j] * w)   #重み付き多項分布の最尤推定
  }
  
  #混合率を推定
  r <- apply(z, 2, sum) / a
  
  #観測データの対数尤度を計算
  phi[phi==0] <- min(phi[phi > 0])
  L <- LLobz(WX, phi, r, const, a, k1)
  LL <- L$LLob   #観測データの対数尤度
  iter <- iter+1   
  dl <- LL-LL1
  LL1 <- LL
  print(LL)
}

#初期値を設定
theta1 <- extraDistr::rdirichlet(1, rep(1, k1))
alpha <- matrix(0.3, nrow=k1, ncol=k1)
diag(alpha) <- 1.5
theta2 <- extraDistr::rdirichlet(k1, alpha)
theta3 <- extraDistr::rdirichlet(k1, rep(0.3, k2))
z1_vec <- as.numeric(rmnom(a, 1, z) %*% 1:k1)


##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 0.1
alpha02 <- 0.1
beta01 <- 1
beta02 <- 1


##パラメータの格納用配列
THETA1 <- matrix(0, nrow=R/keep, ncol=k1)
THETA2 <- array(0, dim=c(k1, k1, R/keep))
THETA3 <- array(0, dim=c(k1, k2, R/keep))
PHI <- array(0, dim=c(k2, v, R/keep))
SEG1 <- matrix(0, nrow=a, ncol=k1)
SEG2 <- matrix(0, nrow=f, ncol=k2)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##MCMC推定用配列
max_time <- max(t_id)
max_word <- max(words)
index_t11 <- which(t_id==1)
index_t21 <- list()
index_t22 <- list()
for(j in 2:max_word){
  index_t21[[j]] <- which(t_id==j)-1
  index_t22[[j]] <- which(t_id==j)
}

#対数尤度の基準値
LLst <- sum(WX %*% log(colSums(WX)/f))


####ギブスサンプリングでHTMモデルのパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとに文章単位でトピックを生成
  #トピック尤度を計算
  z1_indicate <- rep(z1_vec, w)
  word_par <- matrix(0, nrow=f, ncol=k2)
  for(j in 1:k2){
    word_par[, j] <- theta3[z1_indicate, j] * phi[j, wd]   #トピック尤度
  }
  

  #トピックの割当確率からトピックを生成
  topic_rate <- word_par / rowSums(word_par)
  Zi2 <- rmnom(f, 1, topic_rate)
  z2_vec <- as.numeric(Zi2 %*% 1:k2)


  ##HMMで文章単位のセグメントを生成
  #文章単位でのトピック頻度行列を作成
  HMM_data <- matrix(0, nrow=a, ncol=k2)
  for(i in 1:a){
    HMM_data[i, ] <- vec_list[[i]] %*% Zi2[s_list[[i]], , drop=FALSE]
  }
  
  #潜在変数ごとに尤度を推定
  theta_log <- log(t(theta3))
  LLi0 <- HMM_data %*% theta_log   #対数尤度
  LLi_max <- rowMaxs(LLi0)
  LLi <- exp(LLi0 - LLi_max)   #尤度に変換
  
  #セグメント割当確率の推定とセグメントの生成
  z_rate1 <- matrix(0, nrow=a, ncol=k1)
  Zi1 <- matrix(0, nrow=a, ncol=k1)
  z1_vec <- rep(0, a)
  rf02 <- matrix(0, nrow=k1, ncol=k1) 

  for(j in 1:max_time){
    if(j==1){
      #セグメントの割当確率
      LLs <- matrix(theta1, nrow=length(index_t11), ncol=k1, byrow=T) * LLi[index_t11, ]   #重み付き尤度
      z_rate1[index_t11, ] <- LLs / rowSums(LLs)   #割当確率
      
      #多項分布よりセグメントを生成
      Zi1[index_t11, ] <- rmnom(length(index_t11), 1, z_rate1[index_t11, ])
      z1_vec[index_t11] <- as.numeric(Zi1[index_t11, ] %*% 1:k1)
      
      #混合率のパラメータを更新
      rf01 <- colSums(Zi1[index_t11, ])
      
    } else {
      
      #セグメントの割当確率
      index <- index_t22[[j]]
      LLs <- theta2[z1_vec[index_t21[[j]]], , drop=FALSE] * LLi[index, , drop=FALSE]   #重み付き尤度
      z_rate1[index, ] <- LLs / rowSums(LLs)   #割当確率
      
      #多項分布よりセグメントを生成
      Zi1[index, ] <- rmnom(length(index), 1, z_rate1[index, ])
      z1_vec[index] <- as.numeric(Zi1[index, ] %*% 1:k1)
      
      #混合率のパラメータを更新
      rf02 <- rf02 + t(Zi1[index_t21[[j]], , drop=FALSE]) %*% Zi1[index, , drop=FALSE]   #マルコフ推移
    }
  }
  
  ##パラメータをサンプリング
  #ディクレリ分布からHMMの混合率をサンプリング
  rf11 <- colSums(Zi1[index_t11, ]) + beta01
  rf12 <- rf02 + alpha01
  theta1 <- extraDistr::rdirichlet(1, rf11)
  theta2 <- extraDistr::rdirichlet(k1, rf12)

  #トピック分布のパラメータをサンプリング
  wf0 <- matrix(0, nrow=k1, ncol=k2)
  for(j in 1:k1){
    wf0[j, ] <- colSums(HMM_data * Zi1[, j])
  }
  wf <- wf0 + beta02
  theta3 <- extraDistr::rdirichlet(k1, wf)
  
  #単語分布phiをサンプリング
  vf0 <- matrix(0, nrow=k2, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi2[word_list[[j]], , drop=FALSE])
  }
  vf <- vf0 + alpha02
  phi <- extraDistr::rdirichlet(k2, vf)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[mkeep, ] <- theta1
    THETA2[, , mkeep] <- theta2
    THETA3[, , mkeep] <- theta3
    PHI[, , mkeep] <- phi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      print(c(sum(log(rowSums(word_par))), LLst))
      print(round(rbind(theta1, thetat1), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
    }
  }
}


####サンプリング結果の可視化と要約####
burnin <- 2000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#HMMの初期分布のサンプリング結果
matplot(THETA1, type="l", xlab="サンプリング数", ylab="パラメータ")

#HMMのパラメータのサンプリング結果
matplot(t(THETA2[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[15, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#文書のトピック分布のサンプリング結果
matplot(t(THETA3[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA3[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA3[15, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#単語の出現確率のサンプリング結果
matplot(t(PHI[, 1, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[, 100, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[, 200, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[, 300, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[, 400, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[, 500, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")


##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
word <- round(t(rbind(word_mu, phit)), 3)
colnames(word) <- 1:ncol(word)
word

##トピックの事後分布の要約
round(cbind(z1, seg1_mu <- SEG1 / length(burnin:RS)), 3)
round(cbind(z2, seg2_mu <- SEG2 / rowSums(SEG2)), 3)




