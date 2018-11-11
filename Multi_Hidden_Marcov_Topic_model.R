#####Multi Hidden Marcov Topic Model#####
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
k1 <- 7   #HMMの混合数
k2 <- 10   #共通のトピック数
k3 <- 10   #文書のトピック数
d <- 2000   #文書数
v1 <- 300   #文書集合の語彙数
v2 <- 300   #文書特有の語彙数
v <- v1 + v2
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
for(i in 1:a){
  s_list[[i]] <- which(nd_d==i)
}

##パラメータの設定
#ディレクリ分布のパラメータ
alpha01 <- rep(1, k1)
alpha02 <- matrix(0.3, nrow=k1, ncol=k1)
diag(alpha02) <- 1.5
alpha03 <- rep(0.2, k2)
alpha04 <- rep(0.2, k3)
alpha11 <- c(rep(0.1, v1), rep(0.00025, v2))
alpha12 <- c(rep(0.00025, v1), rep(0.1, v2))

for(l in 1:100){
  print(l)
  #パラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(1, alpha01)
  theta2 <- thetat2 <- extraDistr::rdirichlet(k1, alpha02)
  theta3 <- thetat3 <- extraDistr::rdirichlet(k1, alpha03) 
  theta4 <- thetat4 <- extraDistr::rdirichlet(d, alpha04)
  gamma <- gammat <- extraDistr::rdirichlet(k2, alpha11)
  phi <- phit <- extraDistr::rdirichlet(k3, alpha12)
  omega <- omegat <- rbeta(d, 25.0, 27.5)
  
  ##モデルにもとづき単語を生成する
  wd_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  ID_list <- list()
  td_list <- list()
  Z1_list <- list()
  Z2_list <- list()
  Z3_list <- list()
  Z4_list <- list()
  
  #単語ごとに文章共通か文章固有かを生成
  z1_vec <- rbinom(f, 1, omega[ID_d])
  
  #文章ごとのセグメントを生成
  z2_vec2 <- z2_vec <- as.numeric(rmnom(a, 1, theta1) %*% 1:k1)
  freq <- c()
  
  for(i in 1:a){
    flag <- sum(1-z1_vec[s_list[[i]]])
    freq <- c(freq, flag)
    if(t_id[i]==1){
      if(flag==0){
       z2_vec[i] <- 0 
       next
      } 
      if(flag > 0){
        next
      }
    }
    if(t_id[i]!=0){
      if(z2_vec[i-1]==0 & flag==0){
        z2_vec[i] <- 0
        next
      }
      if(z2_vec[i-1]==0 & flag > 0){
        next
      }
      if(z2_vec[i-1]!=0 & flag > 0){
        z2 <- rmnom(1, 1, theta2[z2_vec[i-1], ])
        z2_vec[i] <- as.numeric(z2 %*% 1:k1)
        next
      }
      if(z2_vec[i-1]!=0 & flag==0){
        z2_vec[i] <- z2_vec[i-1]
      }
    }
  }
  Z1 <- z1_vec
  Z2 <- z2_vec
  
  #文章ごとに逐次的にトピックと単語を生成
  Z3_list <- list()
  Z4_list <- list()
  wd_list <- list()
  WX <- matrix(0, nrow=a, ncol=v)
  
  for(i in 1:a){
    if(i%%1000==0){
      print(i)
    }
    #パラメータの格納用配列
    index_id <- u_id[i]
    n <- w[i]
    z3_vec <- rep(0, n)
    z4_vec <- rep(0, n)
    wd <- rep(0, n)
    
    #文書共通か文書固有かの指示変数を取り出す
    flag <- z1_vec[s_list[[i]]]
    index1 <- which(flag==0)
    index2 <- which(flag==1)
    
    #文章共通のトピックを生成
    if(length(index1) > 0){
      z3 <- rmnom(length(index1), 1, theta3[z2_vec[i], ]) 
      z3_vec[index1] <- as.numeric(z3 %*% 1:k2)
    }
  
    #文書固有のトピックの生成
    if(length(index2) > 0){
      z4 <- rmnom(length(index2), 1, theta4[index_id, ]) 
      z4_vec[index2] <- as.numeric(z4 %*% 1:k3)
    }
    
    #文章共通のトピックから単語を生成
    index_topic1 <- z3_vec[z3_vec!=0]
    if(length(index_topic1) > 0){
      wd1 <- rmnom(length(index_topic1), 1, gamma[index_topic1, ])
      wd[index1] <- as.numeric(wd1 %*% 1:v)
    }
    
    #文書固有のトピックから単語を生成
    index_topic2 <- z4_vec[z4_vec!=0]
    if(length(index_topic2) > 0){
      wd2 <- rmnom(length(index_topic2), 1, phi[index_topic2, ]) 
      wd[index2] <- as.numeric(wd2 %*% 1:v)
    }
    
    #パラメータを格納
    wd_list[[i]] <- wd 
    WX[i, ] <- colSums(wd1) + colSums(wd2)
    Z3_list[[i]] <- z3_vec
    Z4_list[[i]] <- z4_vec
  }
  
  #リストをベクトル変換
  Z3 <- unlist(Z3_list)
  Z4 <- unlist(Z4_list[[i]])
  wd <- unlist(wd_list)
  if(length(unique(wd))==v){
    break
  }
}
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
  expl <- r_matrix * exp(LLi - LLi_max)
  expl_log <- log(expl)
  expl_max <- matrix(log(max(expl[1, ])), nrow=hh, ncol=k)
  z <- exp(expl_log - (log(rowSums(exp(expl_log - expl_max))) + expl_max))   #セグメント割当確率
  
  #観測データの対数尤度
  r_log <- matrix(log(r), nrow=hh, ncol=k, byrow=T)
  LLosum <- sum(log(rowSums(exp(r_log + LLi))))   #観測データの対数尤度
  rval <- list(LLob=LLosum, z=z, LL=LLi)
  return(rval)
}


####MHMMトピックモデルのMCMCアルゴリズムの設定####
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
theta4 <- thetat4
gamma <- gammat
phi <- phit
omega <- omegat
r <- mean(omegat)
z2_vec <- Z2


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
theta3 <- extraDistr::rdirichlet(k1, rep(0.4, k2))
theta4 <- extraDistr::rdirichlet(d, rep(0.4, k3))
r <- 0.5
gamma <- extraDistr::rdirichlet(k2, c(rep(0.3, v1), rep(0.05, v2)))
phi <- extraDistr::rdirichlet(k3, c(rep(0.05, v1), rep(0.3, v2)))
z2_vec <- as.numeric(rmnom(a, 1, z) %*% 1:k1)


##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 0.01
alpha02 <- 0.01
alpha03 <- 0.01
beta01 <- 1
beta02 <- 1
beta03 <- 1
beta04 <- 1


##パラメータの格納用配列
THETA1 <- matrix(0, nrow=R/keep, ncol=k1)
THETA2 <- array(0, dim=c(k1, k1, R/keep))
THETA3 <- array(0, dim=c(k1, k2, R/keep))
THETA4 <- array(0, dim=c(d, k3, R/keep))
GAMMA <- array(0, dim=c(k2, v, R/keep))
PHI <- array(0, dim=c(k3, v, R/keep))
OMEGA <- rep(0, R/keep)
SEG1 <- rep(0, f)
SEG2 <- matrix(0, nrow=a, ncol=k1)
SEG3 <- matrix(0, nrow=f, ncol=k2)
SEG4 <- matrix(0, nrow=f, ncol=k3)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"
storage.mode(SEG3) <- "integer"
storage.mode(SEG4) <- "integer"


##MCMC推定用配列
max_time <- max(t_id)
max_word <- max(words)
index_t11 <- which(t_id==1)
index_t21 <- list()
index_t22 <- list()
for(j in 2:max_time){
  index_t21[[j]] <- which(t_id==j)-1
  index_t22[[j]] <- which(t_id==j)
}

#基準対数尤度を設定
LLst <- sum(sparse_data %*% log(colSums(WX)/sum(WX)))


####ギブスサンプリングでHTMモデルのパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとに文書共通か文書固有かを生成
  #単語ごとに文書共通のトピック尤度を計算
  z2_indicate <- z2_vec[nd_d]
  index_zeros <- which(z2_indicate==0)
  word_par1 <- matrix(0, nrow=f, ncol=k2)
  if(length(index_zeros) > 0){
    for(j in 1:k2){
      word_par1[-index_zeros, j] <- theta3[z2_indicate[-index_zeros], j] * gamma[j, wd[-index_zeros]]
    }
  } else {
    for(j in 1:k2){
      word_par1[, j] <- theta3[z2_indicate, j] * gamma[j, wd]
    }
  }
  Li1 <- (1-r) * rowSums(word_par1)
  
  #単語ごとに文書固有のトピック尤度を計算
  word_par2 <- burden_fr(theta4, phi, wd, words, k3)
  Li2 <- r * rowSums(word_par2$Bur)

  #潜在確率の計算と潜在変数の生成
  z_rate1 <- Li2 / (Li1+Li2)
  Zi1 <- rbinom(f, 1, z_rate1)
  index_z1 <- which(Zi1==1)
  
  #ベータ分布から混合率の更新
  r <- rbeta(1, sum(Zi1)+beta01, sum(1-Zi1)+beta01)

  
  ##多項分布から単語トピックを生成
  #文書共通のトピックを生成
  Zi3 <- matrix(0, nrow=f, ncol=k2)
  z_rate3 <- word_par1 / rowSums(word_par1)
  Zi3[-index_z1, ] <- rmnom(f-length(index_z1), 1, z_rate3[-index_z1, ])
  
  #文書固有のトピックを生成
  Zi4 <- matrix(0, nrow=f, ncol=k3)
  z_rate4 <- word_par2$Br
  Zi4[index_z1, ] <- rmnom(length(index_z1), 1, z_rate4[index_z1, ])
  
  
  ##HMMで文章単位のセグメントを生成
  #文章単位でのトピック頻度行列を作成
  HMM_data <- matrix(0, nrow=a, ncol=k2)
  for(i in 1:a){
    HMM_data[i, ] <- rep(1, length(s_list[[i]])) %*% Zi3[s_list[[i]], , drop=FALSE]
  }
  
  #潜在変数ごとに尤度を推定
  theta_log <- log(t(theta3))
  LLi0 <- HMM_data %*% theta_log   #対数尤度
  LLi_max <- apply(LLi0, 1, max)
  LLi <- exp(LLi0 - LLi_max)   #尤度
  
  #セグメント割当確率の推定とセグメントの生成
  z_rate2 <- matrix(0, nrow=a, ncol=k1)
  Zi2 <- matrix(0, nrow=a, ncol=k1)
  z2_vec <- rep(0, a)
  rf02 <- matrix(0, nrow=k1, ncol=k1) 
  
  for(j in 1:max_time){
    if(j==1){
      #セグメントの割当確率
      LLs <- matrix(theta1, nrow=length(index_t11), ncol=k1, byrow=T) * LLi[index_t11, ]   #重み付き尤度
      z_rate2[index_t11, ] <- LLs / rowSums(LLs)   #割当確率
      
      #多項分布よりセグメントを生成
      Zi2[index_t11, ] <- rmnom(length(index_t11), 1, z_rate2[index_t11, ])
      z2_vec[index_t11] <- as.numeric(Zi2[index_t11, ] %*% 1:k1)
      
      #混合率のパラメータを更新
      rf01 <- colSums(Zi2[index_t11, ])
      
    } else {
      
      #セグメントの割当確率
      index <- index_t22[[j]]
      LLs <- theta2[z2_vec[index_t21[[j]]], , drop=FALSE] * LLi[index, , drop=FALSE]   #重み付き尤度
      z_rate2[index, ] <- LLs / rowSums(LLs)   #割当確率
      
      #多項分布よりセグメントを生成
      Zi2[index, ] <- rmnom(length(index), 1, z_rate2[index, ])
      z2_vec[index] <- as.numeric(Zi2[index, ] %*% 1:k1)
      
      #混合率のパラメータを更新
      rf02 <- rf02 + t(Zi2[index_t21[[j]], , drop=FALSE]) %*% Zi2[index, , drop=FALSE]   #マルコフ推移
    }
  }

  ##パラメータをサンプリング
  #ディクレリ分布からHMMの混合率をサンプリング
  rf11 <- colSums(Zi2[index_t11, ]) + beta01
  rf12 <- rf02 + alpha01
  theta1 <- extraDistr::rdirichlet(1, rf11)
  theta2 <- extraDistr::rdirichlet(k1, rf12)
  
  #文書共通のトピック分布のパラメータをサンプリング
  wf0 <- matrix(0, nrow=k1, ncol=k2)
  for(j in 1:k1){
    wf0[j, ] <- colSums(Zi2[nd_d, j] * Zi3)
  }
  wf <- wf0 + beta01
  theta3 <- extraDistr::rdirichlet(k1, wf)
  
  
  #文書固有のトピック分布のパラメータをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k3)
  for(i in 1:d){
    wsum0[i, ] <- rep(1, length(doc_list[[i]])) %*% Zi4[doc_list[[i]], ]
  }
  wsum <- wsum0 + beta01
  theta4 <- extraDistr::rdirichlet(d, wsum)
  
  #単語分布gammaをサンプリング
  gf0 <- matrix(0, nrow=k2, ncol=v)
  for(j in 1:v){
    gf0[, j] <- colSums(Zi3[word_list[[j]], , drop=FALSE])
  }
  gf <- gf0 + alpha02
  gamma <- extraDistr::rdirichlet(k2, gf)
  
  #単語分布phiをサンプリング
  vf0 <- matrix(0, nrow=k2, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi4[word_list[[j]], , drop=FALSE])
  }
  vf <- vf0 + alpha03
  phi <- extraDistr::rdirichlet(k3, vf)

  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[mkeep, ] <- theta1
    THETA2[, , mkeep] <- theta2
    THETA3[, , mkeep] <- theta3
    THETA4[, , mkeep] <- theta4
    GAMMA[, , mkeep] <- gamma
    PHI[, , mkeep] <- phi
    #OMEGA[mkeep] <- r
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      SEG3 <- SEG3 + Zi3
      SEG4 <- SEG4 + Zi4
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      gamma[gamma==0] <- min(gamma[gamma!=0])
      phi[phi==0] <- min(phi[phi!=0])
      LL <- sum(sparse_data %*% t(log(gamma)) * (1-Zi1) * Zi3) + sum(sparse_data %*% t(log(phi)) * Zi1 * Zi4)
      print(rp)
      print(c(LL, LLst))
      print(round(c(mean(r), mean(omegat)), 3))
      print(round(cbind(theta2, thetat2), 3))
      print(round(cbind(theta3, thetat3), 3))
      print(round(cbind(phi[, 296:305], phit[, 296:305]), 3))
    }
  }
}


####サンプリング結果の可視化と要約####
burnin <- 1000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(THETA1, type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[7, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA3[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA3[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA3[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA3[7, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA4[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA4[100, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA4[1000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA4[2000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#単語の出現確率のサンプリング結果
matplot(t(PHI[, 1, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[, 200, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
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





