#####多変量混合LDAモデル#####
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

#set.seed(2578)

####データの発生####
##データの設定
k1 <- 7   #文章のセグメント数
k2 <- 7   #文書単位ののトピック数
k3 <- 8   #セグメント単位のトピック数 
d <- 2000   #文書数
v1 <- 200   #文書集合全体に共通する語彙数
v2 <- 300   #文書単体に共通する語彙数 
v <- v1 + v2
s <- rpois(d, 11.5)   #文章数
s[s < 5] <- ceiling(runif(sum(s < 5), 5, 10))
a <- sum(s)   #総文章数
w1 <- rpois(a, 5.5)   
w1[w1 < 2] <- ceiling(runif(sum(w1 < 2), 2, 5))
w2 <- rpois(a, 10.5)   #文章あたりの単語数
w2[w2 < 3] <- ceiling(runif(sum(w2 < 3), 3, 10))
f1 <- sum(w1)   #総単語数
f2 <- sum(w2) 
f <- f1 + f2

#文書IDの設定
u_id <- rep(1:d, s)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:s[i])}
words1 <- as.numeric(tapply(w1, u_id, sum))
words2 <- as.numeric(tapply(w2, u_id, sum))
words <- words1 + words2

##パラメータを設定
#ディレクリ分布のパラメータ
alpha01 <- seq(3.0, 0.2, length=k1*5)[((1:(k1*5))%%5)==0]
alpha02 <- matrix(0.3, nrow=k1, ncol=k1)
diag(alpha02) <- 2.5
alpha03 <- rep(0.25, k2)
alpha04 <- rep(0.3, k3)
alpha11 <- rep(0.1, v1)
alpha12 <- rep(0.4, v2)
alpha13 <- rep(0.05, v2)

#ディレクリ分布よりパラメータを生成
omegat <- omega <- extraDistr::rdirichlet(1, alpha01)   #文書の先頭トピック
gammat <- gamma <- extraDistr::rdirichlet(k1, alpha02)   #文書セグメント単位のトピック分布
thetat1 <- theta1 <- extraDistr::rdirichlet(d, alpha03)   #文書単位のトピック分布
thetat2 <- theta2 <- extraDistr::rdirichlet(k1, alpha04)   #セグメント単位のトピック分布
psit <- psi <- extraDistr::rdirichlet(k1, alpha11)
phit <- phi <- extraDistr::rdirichlet(k2, alpha12)
tau <- taut <- extraDistr::rdirichlet(k3, alpha13)
betat <- beta <- rbeta(d, 25, 17.5)


##文章ごとに単語を生成する
WX1 <- matrix(0, nrow=a, ncol=v1)
WX2 <- matrix(0, nrow=a, ncol=v2)
y_list <- list()
Z1_list <- list()
Z2_list <- list()
Z3_list <- list()

for(i in 1:d){
  if(i%%100==0){
    print(i)
  }
  z1_vec <- rep(0, s[i])
  
  for(j in 1:s[i]){
    ##単語ごとに生成過程を決定
    index <- which(u_id==i)[j]
    
    ##文章ごとにトピックを生成
    freq1 <- w1[index]
    freq2 <- w2[index]
    
    if(j==1){
      z1 <- rmnom(1, 1, omega)
      z1_vec[j] <- as.numeric(z1 %*% 1:k1)
    } else {
      z1 <- rmnom(1, 1, gamma[z1_vec[j-1], ])
      z1_vec[j] <- as.numeric(z1 %*% 1:k1)
    }
    
    ##単語ごとにトピックを生成
    y <- rbinom(freq2, 1, beta[i])
    z2_vec <- rep(0, freq2)
    z3_vec <- rep(0, freq2)
    sum(1-y)
    
    if(sum(y) > 0){
      z2 <- rmnom(sum(y), 1, theta1[i, ])
      z2_vec[y==1] <- as.numeric(z2 %*% 1:k2)
    }
    if(sum(1-y) > 0){
      z3 <- rmnom(sum(1-y), 1, theta2[z1_vec[j], ])
      z3_vec[y==0] <- as.numeric(z3 %*% 1:k3)
    }
    
    ##トピック分布に基づき単語を生成
    #文章単位での単語を生成
    wn1 <- rep(0, v1)
    wn1 <- colSums(rmnom(freq1, 1, psi[z1_vec[j], ]))
    
    #単語単位での単語を生成
    wn2 <- rep(0, v2)
    if(sum(y) > 0){
      wn2 <- colSums(rmnom(sum(y), 1, phi[z2_vec[z2_vec!=0], ]))
    }
    wn3 <- rep(0, v2)
    if(sum(1-y) > 0){
      wn3 <- colSums(rmnom(sum(1-y), 1, tau[z3_vec[z3_vec!=0], ]))
    }
    
    #生成した単語を格納
    WX1[index, ] <- wn1
    WX2[index, ] <- wn2 + wn3
    if(is.na(sum(WX2[index, ]))==TRUE) {break}
    
    #パラメータを格納
    Z2_list[[index]] <- z2_vec
    Z3_list[[index]] <- z3_vec
    y_list[[index]] <- y
  }
  Z1_list[[i]] <- z1_vec
}

#単語出現がない単語を削除
index_zeros <- which(colSums(WX1)==0)
WX1 <- WX1[, -index_zeros]
psi <- psit <- psi[, -index_zeros]
v1 <- ncol(WX1)

#リスト形式を変換
Y <- unlist(y_list)
z1 <- unlist(Z1_list)
z2 <- unlist(Z2_list)
z3 <- unlist(Z3_list)
WX <- cbind(WX1, WX2)


####トピックモデル推定のためのデータと関数の準備####
##それぞれの文書中の単語の出現および補助情報の出現をベクトルに並べる
##データ推定用IDを作成
ID_list <- list()
td1_list <- list()
td2_list <- list()
wd_list <- list()

#文書idごとに文章idおよび単語idを作成
for(i in 1:a){
  
  #文書IDを記録
  ID_list[[i]] <- rep(u_id[i], w2[i])
  td1_list[[i]] <- rep(i, w2[i])
  td2_list[[i]] <- rep(t_id[i], w2[i])
  
  #単語IDを記録
  num1 <- WX2[i, ] * 1:v2
  num2 <- which(num1 > 0)
  W1 <- WX2[i, (WX2[i, ] > 0)]
  number <- rep(num2, W1)
  wd_list[[i]] <- number
}

#リストをベクトルに変換
ID_d <- unlist(ID_list)
td1_d <- unlist(td1_list)
td2_d <- unlist(td2_list)
wd <- unlist(wd_list)

##インデックスを作成
doc_list <- list()
id_list <- list()
sent_list <- list()
word_list <- list()
for(i in 1:d){doc_list[[i]] <- which(ID_d==i)}
for(i in 1:d){id_list[[i]] <- which(u_id==i)}
for(i in 1:a){sent_list[[i]] <- which(td1_d==i)}
for(i in 1:v2){word_list[[i]] <- which(wd==i)}


####マルコフ連鎖モンテカルロ法でLDAモデルを推定####
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

#対数尤度の目標値
LLst1 <- sum(dmnom(WX1, rowSums(WX1), colSums(WX1)/sum(WX1), log=TRUE))
LLst2 <- sum(WX2 %*% log(colSums(WX2)/sum(WX2)))
LLst <- LLst1 + LLst2


##アルゴリズムの設定
R <- 10000
keep <- 2  
iter <- 0
burnin <- 2000/keep
disp <- 10

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 1 
alpha02 <- 1
alpha03 <- 1
beta01 <- 0.5
beta02 <- 0.5
beta03 <- 0.5

##パラメータの初期値
theta1 <- thetat1
theta2 <- thetat2
rt0 <- r0 <- as.numeric(omega)
rt1 <- r1 <- gammat
rt2 <- r2 <- beta[ID_d]
psi <- psit
phi <- phit
tau <- taut

#tfidfで初期値を設定
tf11 <- colMeans(WX1)*100+1
tf21 <- colMeans(WX2)*10
idf21 <- log(nrow(WX2)/colSums(WX2 > 0))
idf22 <- log(nrow(WX2)/colSums(WX2==0))

#単語トピック単位のパラメータの初期値
theta1 <- extraDistr::rdirichlet(d, rep(1, k2))   #文書単位のトピックの初期値
theta2 <- extraDistr::rdirichlet(k1, rep(1, k3))   #セグメント単位のトピックの初期値
psi <- extraDistr::rdirichlet(k1, tf11)   #文章単位の単語出現確率の初期値
phi <- extraDistr::rdirichlet(k2, tf21)   #文書単位の単語出現確率の初期値
tau <- extraDistr::rdirichlet(k3, tf21)   #セグメント単位の単語出現確率の初期値
r0 <- rep(1/k1, k1)
par <- matrix(0.3, nrow=k1, ncol=k1)
diag(par) <- 2.0
r1 <- extraDistr::rdirichlet(k1, par)
r2 <- rep(0.5, f2)


##パラメータの格納用配列
THETA1 <- array(0, dim=c(d, k2, R/keep))
THETA2 <- array(0, dim=c(k1, k3, R/keep))
R0 <- matrix(0, nrow=R/keep, ncol=k1)
R1 <- array(0, dim=c(k1, k1, R/keep))
PSI <- array(0, dim=c(k1, v1, R/keep))
PHI <- array(0, dim=c(k2, v2, R/keep))
TAU <- array(0, dim=c(k3, v2, R/keep))
SEG1 <- matrix(0, nrow=a, ncol=k1)
SEG2 <- matrix(0, nrow=f2, ncol=k2)
SEG3 <- matrix(0, nrow=f2, ncol=k3)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"
storage.mode(SEG3) <- "integer"

##MCMC推定用配列
max_time <- max(t_id)
index_t11 <- which(t_id==1)
index_t21 <- list()
index_t22 <- list()
for(j in 2:max_time){
  index_t21[[j]] <- which(t_id==j)-1
  index_t22[[j]] <- which(t_id==j)
}
wx1_const <- lfactorial(w1) - rowSums(lfactorial(WX1))   #多項分布の密度関数の対数尤度の定数

####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##文章単位のトピックをサンプリング
  #セグメントごとの尤度を推定
  psi_log <- log(t(psi))
  LLi0 <- wx1_const + WX1 %*% psi_log 
  LLi_max <- apply(LLi0, 1, max)
  LLi <- exp(LLi0 - LLi_max)
  
  #セグメント割当確率の推定とセグメントの生成
  z1_rate <- matrix(0, nrow=a, ncol=k1)
  Zi1 <- matrix(0, nrow=a, ncol=k1)
  z1_vec <- rep(0, a)
  rf02 <- matrix(0, nrow=k1, ncol=k2) 
  
  for(j in 1:max_time){
    if(j==1){
      #セグメントの割当確率
      LLs <- matrix(r0, nrow=length(index_t11), ncol=k1, byrow=T) * LLi[index_t11, ]   #重み付き尤度
      z1_rate[index_t11, ] <- LLs / rowSums(LLs)   #割当確率
      
      #多項分布よりセグメントを生成
      Zi1[index_t11, ] <- rmnom(length(index_t11), 1, z1_rate[index_t11, ])
      z1_vec[index_t11] <- as.numeric(Zi1[index_t11, ] %*% 1:k1)
      
      #混合率のパラメータを更新
      rf01 <- colSums(Zi1[index_t11, ])
      
    } else {
      
      #セグメントの割当確率
      index <- index_t22[[j]]
      z1_vec[index_t21[[j]]]
      LLs <- r1[z1_vec[index_t21[[j]]], , drop=FALSE] * LLi[index, , drop=FALSE]   #重み付き尤度
      z1_rate[index, ] <- LLs / rowSums(LLs)   #割当確率
      
      #多項分布よりセグメントを生成
      Zi1[index, ] <- rmnom(length(index), 1, z1_rate[index, ])
      z1_vec[index] <- as.numeric(Zi1[index, ] %*% 1:k1)
      
      #混合率のパラメータを更新
      rf02 <- rf02 + t(Zi1[index_t21[[j]], , drop=FALSE]) %*% Zi1[index, , drop=FALSE]   #マルコフ推移
    }
  }
  
  #ディクレリ分布から混合率をサンプリング
  rf11 <- colSums(Zi1[index_t11, ]) + alpha01
  rf12 <- rf02 + alpha01
  r0 <- extraDistr::rdirichlet(1, rf11)
  r1 <- extraDistr::rdirichlet(k1, rf12)
  
  #単語分布psiをサンプリング
  df0 <- matrix(0, nrow=k1, ncol=v1)
  for(j in 1:k1){
    df0[j, ] <- colSums(WX1 * Zi1[, j])
  }
  df <- df0 + alpha01
  psi <- extraDistr::rdirichlet(k1, df)
  
  
  ##単語ごとにトピックの生成過程をサンプリング
  #文書特有のトピック分布のパラメータを推定
  word_par1 <- burden_fr(theta1, phi, wd, words2, k2)
  LLw1 <- rowSums(word_par1$Bur)
  
  #セグメント特有のトピック分布のパラメータを推定
  word_par2 <- matrix(0, nrow=f2, ncol=k3)
  par2 <- theta2[rep(z1_vec, w2), ]
  for(j in 1:k3){
    word_par2[, j] <- par2[, j] * tau[j, wd]
  }
  LLw2 <- rowSums(word_par2)
  
  #スイッチング変数を生成
  switching_rate <- r2*LLw1 / (r2*LLw1 + (1-r2)*LLw2)
  y <- rbinom(f2, 1, switching_rate)
  index_y <- which(y==1)
  
  #混合率を更新
  par <- as.numeric(tapply(y, ID_d, sum))
  r02 <- rbeta(d, par+beta01, words2-par+beta01)
  r2 <- r02[ID_d]

  ##単語ごとにトピックをサンプリング
  #文書特有の単語トピックをサンプリング
  n <- length(index_y)
  Zi2 <- matrix(0, nrow=f2, ncol=k2)
  Zi2[index_y, ] <- rmnom(n, 1, word_par1$Br[index_y, ])
  z2_vec <- as.numeric(Zi2 %*% 1:k2)
  
  #セグメント特有の単語トピックをサンプリング
  word_rate2 <- word_par2 / rowSums(word_par2)   #トピックの割当確率
  Zi3 <- matrix(0, nrow=f2, ncol=k3)
  Zi3[-index_y, ] <- rmnom(f2-n, 1, word_rate2[-index_y, ])
  z3_vec <- as.numeric(Zi3 %*% 1:k3)

  
  ##パラメータをサンプリング
  #文書特有のトピック分布theta1をサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k2)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi2[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha02
  theta1 <- extraDistr::rdirichlet(d, wsum)
  
  #セグメント特有のトピック分布theta2をサンプリング
  dsum0 <- matrix(0, nrow=k1, ncol=k3)
  for(j in 1:k1){
    dsum0[j, ] <- colSums(Zi3 * rep(Zi1[, j], w2))
  }
  dsum <- dsum0 + alpha03
  theta2 <- extraDistr::rdirichlet(k1, dsum)

  
  #単語分布phiおよびtauをサンプリング
  vf0 <- matrix(0, nrow=k2, ncol=v2)
  tf0 <- matrix(0, nrow=k3, ncol=v2)
  for(j in 1:v2){
    vf0[, j] <- colSums(Zi2[word_list[[j]], ])
    tf0[, j] <- colSums(Zi3[word_list[[j]], ])
  }
  vf <- vf0 + beta01
  tf <- tf0 + beta02
  phi <- extraDistr::rdirichlet(k2, vf)
  tau <- extraDistr::rdirichlet(k3, tf)

  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    R0[mkeep, ] <- r0
    R1[, , mkeep] <- r1
    PSI[, , mkeep] <- psi
    PHI[, , mkeep] <- phi
    TAU[, , mkeep] <- tau
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      SEG3 <- SEG3 + Zi3
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      LL1 <- sum(Zi1*LLi0)
      LL02 <- matrix(0, nrow=sum(y), ncol=k2)
      for(j in 1:k2){
        LL02[, j] <- phi[j, wd[index_y]]
      }
      LL2 <- sum(log(rowSums(LL02 * Zi2[index_y, ])))
      LL03 <- matrix(0, nrow=sum(1-y), ncol=k3)
      for(j in 1:k3){
        LL03[, j] <- tau[j, wd[-index_y]]
      }
      LL3 <- sum(log(rowSums(LL03 * Zi3[-index_y, ])))
      
      print(rp)
      print(round(c(mean(r02), mean(betat)), 3))
      print(c(LL1+LL2+LL3, LLst))
      print(round(cbind(theta1[1:7, ], thetat1[1:7, ]), 3))
      print(round(cbind(theta2[1:7, ], thetat2[1:7, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
    }
  }
}


####サンプリング結果の可視化と要約####
burnin <- 2000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA1[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[100, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[1000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[2000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[7, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

matplot(t(GAMMA[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(GAMMA[2, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(GAMMA[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(GAMMA[4, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, 1:10, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[2, 51:60, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[3, 101:110, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[4, 151:160, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(PSI[1, 1:10, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PSI[2, 51:60, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PSI[3, 101:110, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PSI[4, 151:160, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")

#一般語の出現確率のサンプリング結果
matplot(GAMMA[, 286:295], type="l", ylab="パラメータ", main="単語の出現率のサンプリング結果")
matplot(GAMMA[, 296:305], type="l", ylab="パラメータ", main="単語の出現率のサンプリング結果")
matplot(GAMMA[, 306:315], type="l", ylab="パラメータ", main="単語の出現率のサンプリング結果")


##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu1 <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
word1 <- round(t(rbind(word_mu1, phit)), 3)
word_mu2 <- apply(OMEGA[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
word2 <- round(t(rbind(word_mu2, omegat)), 3)
word <- round(t(rbind(word_mu1, word_mu2, phit, omegat)), 3)
colnames(word) <- 1:ncol(word)

word_mu3 <- apply(GAMMA[burnin:(R/keep), ], 2, mean)   #単語の出現率の事後平均
round(rbind(word_mu3, gamma=gammat), 3)


##トピックの事後分布の要約
round(seg1_mu <- SEG1 / rowSums(SEG1), 3)
round(seg2_mu <- SEG2 / rowSums(SEG2), 3)
