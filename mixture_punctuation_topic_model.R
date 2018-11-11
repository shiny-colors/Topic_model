#####混合LDAモデル#####
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
k1 <- 10   #単語ごとのトピック数
k2 <- 7   #文章のセグメント数
k3 <- 10   #文章ごとのトピック数
d <- 2000   #文書数
v <- 350   #語彙数
s <- rpois(d, 15)   #文章数
s[s < 5] <- ceiling(runif(sum(s < 5), 5, 10))
a <- sum(s)   #総文章数
w <- rpois(a, 13)   #文章あたりの単語数
w[w < 5] <- ceiling(runif(sum(w < 5), 5, 10))
f <- sum(w)   #総単語数

#文書IDの設定
u_id <- rep(1:d, s)
t_id <- c()
for(i in 1:d){t_id <- c(t_id, 1:s[i])}
words <- as.numeric(tapply(w, u_id, sum))


##パラメータを設定
#ディレクリ分布のパラメータ
alpha01 <- rep(4.0, k2)
alpha02 <- rep(0.15, k3)
alpha03 <- rep(0.25, k1)
alpha11 <- rep(0.4, v)
alpha12 <- rep(0.1, v)

#ディレクリ分布よりパラメータを生成
omegat <- omega <- extraDistr::rdirichlet(1, alpha01)   #セグメント割当確率
gammat <- gamma <- extraDistr::rdirichlet(k2, alpha02)   #文書セグメント単位のトピック分布
thetat <- theta <- extraDistr::rdirichlet(d, alpha03)   #文書単位のトピック分布
phit <- phi <- extraDistr::rdirichlet(k1, alpha11)
psit <- psi <- extraDistr::rdirichlet(k3, alpha12)
betat <- beta <- 0.45

##文章ごとに単語を生成する
WX <- matrix(0, nrow=a, ncol=v)
y_list <- list()
Z1_list <- list()
Z2_list <- list()

for(i in 1:a){

  ##文章ごとにトピック分布を生成
  y <- rbinom(1, 1, beta)
  
  ##文章ごとにトピックを生成
  id <- u_id[i]
  if(y==1){
    z2 <- rmnom(w[i], 1, theta[id, ])
    z2_vec <- as.numeric(z2 %*% 1:k1)
    z1 <- rep(0, k2)
  } else {
    z1 <- rmnom(1, 1, omega)
    z1_vec <- as.numeric(z1 %*% 1:k2)
    z2 <- rmnom(w[i], 1, gamma[z1_vec, ])
    z2_vec <- as.numeric(z2 %*% 1:k3)
  }
  
  #トピック分布に基づき単語を生成
  if(y==1){
    wn <- rmnom(w[i], 1, phi[z2_vec, ])
  } else {
    wn <- rmnom(w[i], 1, psi[z2_vec, ])
  }
  WX[i, ] <- colSums(wn)
  
  #パラメータを格納
  Z1_list[[i]] <- z1
  Z2_list[[i]] <- z2_vec
  y_list[[i]] <- y
}

#リスト形式を変換
Z1 <- do.call(rbind, Z1_list)
Z2 <- unlist(Z2_list)
Y <- unlist(y_list)
z2_freq <- as.numeric(table(Z2[Y==0]))


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
  ID_list[[i]] <- rep(u_id[i], w[i])
  td1_list[[i]] <- rep(i, w[i])
  td2_list[[i]] <- rep(t_id[i], w[i])
  
  #単語IDを記録
  num1 <- WX[i, ] * 1:v
  num2 <- which(num1 > 0)
  W1 <- WX[i, (WX[i, ] > 0)]
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
for(i in 1:v){word_list[[i]] <- which(wd==i)}


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


##アルゴリズムの設定
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 1 
alpha02 <- 1
alpha03 <- 25
beta01 <- 0.5


##パラメータの初期値
#tfidfで初期値を設定
tf0 <- colMeans(WX)*10
idf1 <- log(nrow(WX)/colSums(WX > 0))
idf2 <- log(nrow(WX)/colSums(WX==0))

#初期値設定のためにトピックモデルを推定
#単語トピック単位のパラメータの初期値
theta <- extraDistr::rdirichlet(d, rep(10, k1))   #文書単位のトピックの初期値
phi <- extraDistr::rdirichlet(k1, tf0)   #文書単位の出現確率の初期値
gamma <- extraDistr::rdirichlet(k2, rep(10, k3))   #文章セグメントのトピックの初期値
psi <- extraDistr::rdirichlet(k3, rep(0.2, v))   #文章セグメントの出現確率の初期値
r <- c(0.5, rep(0.5/k2, k2))

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k1, R/keep))
PHI <- array(0, dim=c(k1, v, R/keep))
GAMMA <- array(0, dim=c(k2, k3, R/keep))
PSI <- array(0, dim=c(k3, v, R/keep))
OMEGA <- matrix(0, nrow=R/keep, ncol=k2+1)
SEG1 <- matrix(0, nrow=a, ncol=k2+1)
SEG2 <- matrix(0, nrow=f, ncol=k1)
SEG3 <- matrix(0, nrow=f, ncol=k3)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"
storage.mode(SEG3) <- "integer"

##MCMC推定用配列
wsum0 <- matrix(0, nrow=d, ncol=k1)
vf0 <- matrix(0, nrow=k1, ncol=v)
dsum0 <- matrix(0, nrow=d, ncol=k2)
sf0 <- matrix(0, nrow=k2, ncol=v)



####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとのトピックの尤度を推定
  #文書単位でのトピック尤度
  word_par1 <- burden_fr(theta, phi, wd, words, k1)

  #文章単位のトピック尤度
  LLsums <- matrix(0, nrow=f, ncol=k2)
  for(j in 1:k3){
    LLsums <- LLsums + psi[j, wd] * matrix(gamma[, j], nrow=f, ncol=k2, byrow=T)
  }
  
  ##文章単位のトピックを生成
  #文章単位の尤度とトピック割当確率
  LLind <- cbind(rowSums(word_par1$Bur), LLsums)
  LLho <- matrix(0, nrow=a, ncol=k2+1)
  for(j in 1:a){
    LLho[j, ] <- r * colProds(LLind[sent_list[[j]], ])
  }

  #多項分布よりの文章トピックを生成
  switch_rate <- LLho / rowSums(LLho)   #文章トピックの割当確率
  Zi1 <- rmnom(a, 1, switch_rate)
  zi1_vec <- as.numeric(Zi1 %*% 1:(k2+1))
  index_z1 <- which(rep(Zi1[, 1], w)==1)
  
  #混合率を更新
  r0 <- colSums(Zi1)
  r <- as.numeric(extraDistr::rdirichlet(1, r0 + alpha03))
  
  
  ##多項分布より単語トピックを生成
  #文書単位での単語トピックの生成
  Zi2 <- matrix(0, nrow=f, ncol=k1)
  Zi2[index_z1, ] <- rmnom(length(index_z1), 1, word_par1$Br[index_z1, ])
  zi2_vec <- as.numeric(Zi2 %*% 1:k1)

  #文章セグメント単位での単語トピックの尤度
  index_seg <- rep(zi1_vec-1, w)[-index_z1]
  word_vec <- wd[-index_z1]
  n <- length(index_seg)
  word_par2 <- matrix(0, nrow=n, ncol=k3)
  
  for(j in 1:k3){
    word_par2[, j] <- psi[j, word_vec] * gamma[index_seg, j]
  }
  
  #多項分布より単語トピックを生成
  word_rate2 <- word_par2 / rowSums(word_par2)   #潜在トピックの割当確率
  Zi3 <- matrix(0, nrow=f, ncol=k3)
  Zi3[-index_z1, ] <- rmnom(n, 1, word_rate2)
  zi3_vec <- as.numeric(Zi3 %*% 1:k3)
  Zi3_hat <- Zi3[-index_z1, ]


  ##パラメータをサンプリング
  #文書単位のトピック分布thetaをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k1)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi2[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #文章単位のトピック分布gammaをサンプリング
  vsum0 <- matrix(0, nrow=k2, ncol=k3)
  for(j in 1:k2){
    vsum0[j, ] <- colSums(Zi3_hat[index_seg==j, ])
  }
  vsum <- vsum0 + alpha02 
  gamma <- extraDistr::rdirichlet(k2, vsum)
  
  #単語分布phiおよびpsiをサンプリング
  vf0 <- matrix(0, nrow=k1, ncol=v)
  df0 <- matrix(0, nrow=k3, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi2[word_list[[j]], ])
    df0[, j] <- colSums(Zi3[word_list[[j]], ])
  }
  
  vf <- vf0 + alpha01
  df <- df0 + alpha02
  phi <- extraDistr::rdirichlet(k1, vf)
  psi <- extraDistr::rdirichlet(k3, df)
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma
    PSI[, , mkeep] <- psi
    OMEGA[mkeep, ] <- r
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      SEG3 <- SEG3 + Zi3
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      print(rbind(df0=rowSums(df0), z2_freq))
      print(round(rbind(r, r0=c(betat, (1-betat)*omegat)), 3))
      print(round(cbind(theta[1:7, ], thetat[1:7, ]), 3))
      print(round(cbind(psi[, 1:10], psit[, 1:10]), 3))
    }
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA[100, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA[1000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA[2000, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
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
