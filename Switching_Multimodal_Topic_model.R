#####SwitchingマルチモーダルLDAモデル#####
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

#set.seed(5723)

####データの発生####
##データの設定
s <- 2   #データ数
k1 <- 4   #文書1の独立したトピック
k2 <- 4   #文書2の独立したトピック
k3 <- 5   #共通のトピック
k <- k1 + k2 + k3   #総トピック数
d <- 2000   #文書数
v1 <- 100   #データ1に関係のあるトピックの語彙数
v2 <- 100   #データ2に関係のあるトピックの語彙数
v3 <- 100   #共通のトピックに関係のある語彙数
v4 <- 100   #トピックに関係のない語彙数
v <-   v1 + v2 + v3 + v4   #総語彙数
w1 <- rpois(d, rgamma(d, 45, 0.50))   #1文書あたりの単語数
w2 <- rpois(d, rgamma(d, 50, 0.50))
f1 <- sum(w1)
f2 <- sum(w2)

##パラメータの設定
#ディレクリ分布のパラメータを設定
alpha01 <- rep(0.4, k)
alpha11 <- c(rep(0.5, v1), rep(0.001, v2+v3+v4))
alpha12 <- rep(0.001, v)
alpha13 <- c(rep(0.001, v1+v2), rep(0.5, v3), rep(0.001, v4))
alpha14 <- rep(0.001, v)
alpha15 <- c(rep(0.001, v1), rep(0.5, v2), rep(0.001, v3+v4))
alpha16 <- c(rep(0.001, v1+v2), rep(0.5, v3), rep(0.001, v4))
alpha17 <- c(rep(0.01, v1+v2+v3), rep(30, v4))


#パラメータの生成
thetat <- theta <- extraDistr::rdirichlet(d, alpha01)
phit <- phi <- rbind(extraDistr::rdirichlet(k1, alpha11), extraDistr::rdirichlet(k2, alpha12), 
                     extraDistr::rdirichlet(k3, alpha13))
omega <- omegat <- rbind(extraDistr::rdirichlet(k1, alpha14), extraDistr::rdirichlet(k2, alpha15),
                         extraDistr::rdirichlet(k3, alpha16))
gammat <- gamma <- extraDistr::rdirichlet(1, alpha17)
betat <- beta <- rbeta(d, 25, 15)


##多項分布からデータを生成
WX1 <- matrix(0, nrow=d, ncol=v)
WX2 <- matrix(0, nrow=d, ncol=v)
Z1_list <- list()
Z2_list <- list()
y1_list <- list()
y2_list <- list()

for(i in 1:d){
  ##文書1の単語を生成
  #文書1のトピックを生成
  z1 <- rmnom(w1[i], 1, theta[i, ])   #文書のトピック分布を発生
  z1_vec <- as.numeric(z1 %*% c(1:k))   #トピック割当をベクトル化
 
  #一般語かどうかを生成
  y01 <- rbinom(w1[i], 1, beta[i])
  index_y01 <- which(y01==1)

  #トピックから単語を生成
  wn1 <- rmnom(sum(y01), 1, phi[z1_vec[index_y01], ])   #文書1のトピックから単語を生成
  wn2 <- rmnom(1, sum(1-y01), gamma)   #一般語を生成
  wdn <- colSums(wn1) + colSums(wn2)   #単語ごとに合計して1行にまとめる
  
  WX1[i, ] <- wdn
  Z1_list[[i]] <- z1
  y1_list[[i]] <- y01
  
  
  ##文書2の単語を生成
  #文書2のトピックを生成
  z2 <- rmnom(w2[i], 1, theta[i, ])   #文書のトピック分布を発生
  z2_vec <- as.numeric(z2 %*% c(1:k))   #トピック割当をベクトル化
  
  #一般語かどうかを生成
  y02 <- rbinom(w2[i], 1, beta[i])
  index_y02 <- which(y02==1)

  #トピックから単語を生成
  wn1 <- rmnom(sum(y02), 1, omega[z2_vec[index_y02], ])   #文書1のトピックから単語を生成
  wn2 <- rmnom(1, sum(1-y02), gamma)   #一般語を生成
  wdn <- colSums(wn1) + colSums(wn2)   #単語ごとに合計して1行にまとめる
  WX2[i, ] <- wdn
  Z2_list[[i]] <- z2
  y2_list[[i]] <- y02
}

#リスト形式を変換
Z1 <- do.call(rbind, Z1_list)
y1_vec <- unlist(y1_list)
Z2 <- do.call(rbind, Z2_list)
y2_vec <- unlist(y2_list)


####トピックモデル推定のためのデータと関数の準備####
##データ推定用IDを作成
ID1_list <- list()
wd1_list <- list()
ID2_list <- list()
wd2_list <- list()

#求人ごとに求人IDおよび単語IDを作成
for(i in 1:d){
  print(i)
  
  #文書1の単語のIDベクトルを作成
  ID1_list[[i]] <- rep(i, w1[i])
  num1 <- (WX1[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
  W1 <- WX1[i, (WX1[i, ] > 0)]
  number <- rep(num2, W1)
  wd1_list[[i]] <- number
  
  #文書2の単語のIDベクトルを作成
  ID2_list[[i]] <- rep(i, w2[i])
  num1 <- (WX2[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
  W1 <- WX2[i, (WX2[i, ] > 0)]
  number <- rep(num2, W1)
  wd2_list[[i]] <- number
}

#リストをベクトルに変換
ID1_d <- unlist(ID1_list)
wd1 <- unlist(wd1_list)
ID2_d <- unlist(ID2_list)
wd2 <- unlist(wd2_list)
wd <- c(wd1, wd2)

##インデックスを作成
doc1_list <- list()
word1_list <- list()
doc2_list <- list()
word2_list <- list()
word_list <- list()

for(i in 1:length(unique(ID1_d))) {doc1_list[[i]] <- which(ID1_d==i)}
for(i in 1:v) {word1_list[[i]] <- which(wd1==i)}
for(i in 1:length(unique(ID2_d))) {doc2_list[[i]] <- which(ID2_d==i)}
for(i in 1:v) {word2_list[[i]] <- which(wd2==i)}
for(i in 1:v) {word_list[[i]] <- which(wd==i)}
gc(); gc()


####マルコフ連鎖モンテカルロ法で対応トピックモデルを推定####
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
R <- 10000   #サンプリング回数
keep <- 2   #2回に1回の割合でサンプリング結果を格納
iter <- 0
disp <- 20
burnin <- 1000/keep

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 1.0
beta01 <- 0.25
beta02 <- 0.25
beta03 <- c(f1/20, f1/20)
beta04 <- c(f2/20, f2/20)


##パラメータの初期値
#tfidfで初期値を設定
idf11 <- log(nrow(WX1)/colSums(rbind(WX1, 1) > 0))
idf12 <- log(nrow(WX1)/colSums(rbind(WX1, 1)==0))
idf21 <- log(nrow(WX2)/colSums(rbind(WX2, 1) > 0))
idf22 <- log(nrow(WX2)/colSums(rbind(WX2, 1)==0))


theta <- extraDistr::rdirichlet(d, rep(1, k))   #文書トピックのパラメータの初期値
phi <- extraDistr::rdirichlet(k, idf11*10)   #文書1の単語トピックのパラメータの初期値
omega <- extraDistr::rdirichlet(k, idf21*10)   # 文書2の単語トピックのパラメータの初期値
gamma <- extraDistr::rdirichlet(1, 1/(idf11+idf21)*10)   #一般語のパラメータの初期値
r1<- 0.5; r2 <- 0.5   #混合率の初期値


##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
OMEGA <- array(0, dim=c(k, v, R/keep))
GAMMA <- matrix(0, nrow=R/keep, v)
SEG1 <- matrix(0, nrow=f1, ncol=k)
SEG2 <- matrix(0, nrow=f2, ncol=k)
Y1 <- rep(0, f1)
Y2 <- rep(0, f2)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"
gc(); gc()


##MCMC推定用配列
wsum0 <- matrix(0, nrow=d, ncol=k)
vf0 <- matrix(0, nrow=k, ncol=v)
wf0 <- matrix(0, nrow=k, ncol=v) 
df0 <- rep(0, v)


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##文書1の単語トピックをサンプリング
  #トピックの出現率と尤度を推定
  out1 <- burden_fr(theta, phi, wd1, w1, k)
  LH1 <- out1$Bur
  word_rate1 <- out1$Br
  
  ##文書2の単語トピックをサンプリング
  #トピックの出現率と尤度を推定
  out2 <- burden_fr(theta, omega, wd2, w2, k)
  LH2 <- out2$Bur
  word_rate2 <- out2$Br
  
  #無関係のトピックの出現率と尤度を推定
  LH01 <- gamma[wd1]
  LH02 <- gamma[wd2]

  ##一般語かどうかをサンプリング
  #文書1のスイッチング変数をサンプリング
  Bur11 <- r1 * rowSums(LH1)
  Bur12 <- (1-r1) * LH01
  switch_rate1 <- Bur11 / (Bur11 + Bur12)
  y1 <- rbinom(f1, 1, switch_rate1)
  index_y1 <- which(y1==1)
  
  
  #文書2のスイッチング変数をサンプリング
  Bur21 <- r2 * rowSums(LH2)
  Bur22 <- (1-r2) * LH02
  switch_rate2 <- Bur21 / (Bur21 + Bur22)
  y2 <- rbinom(f2, 1, switch_rate2)
  index_y2 <- which(y2==1)
  
  #ベータ分布から混合率を更新
  par1 <- sum(y1); par2 <- sum(y2)
  r1 <- rbeta(1, par1+beta03[1], f1-par1+beta03[2])
  r2 <- rbeta(1, par2+beta04[1], f2-par2+beta04[2])
  
  ##多項分布から単語トピックをサンプリング
  #文書1のトピックをサンプリング
  Zi1 <- rmnom(f1, 1, word_rate1)   
  Zi1[-index_y1, ] <- 0
  z1_vec <- as.numeric(Zi1 %*% 1:k)
  
  #文書2のトピックをサンプリング
  Zi2 <- rmnom(f2, 1, word_rate2)   
  Zi2[-index_y2, ] <- 0
  z12_vec <- as.numeric(Zi2 %*% 1:k)
  
  ##パラメータをサンプリング
  #トピック分布をディクレリ分布からサンプリング
  wsum01 <- matrix(0, nrow=d, ncol=k)
  wsum02 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum01[i, ] <- colSums(Zi1[doc1_list[[i]], ])
    wsum02[i, ] <- colSums(Zi2[doc2_list[[i]], ])
  }
  wsum <- wsum01  + alpha01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #トピック語の分布をディクレリ分布からサンプリング
  vf0 <- matrix(0, nrow=k, ncol=v)
  wf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi1[word1_list[[j]], , drop=FALSE])
    wf0[, j] <- colSums(Zi2[word2_list[[j]], , drop=FALSE])
  }
  
  vf <- vf0 + beta01
  wf <- wf0 + beta01
  phi <- extraDistr::rdirichlet(k, vf)
  omega <- extraDistr::rdirichlet(k, wf)
  
  #一般語の分布をディクレリ分布からサンプリング
  y1_zeros <- 1-y1
  y2_zeros <- 1-y2
  df01 <- rep(0, v)
  df02 <- rep(0, v)
 
  for(j in 1:v){
    df01[j] <- sum(y1_zeros[word1_list[[j]]])
    df02[j] <- sum(y2_zeros[word2_list[[j]]])
  }
  df1 <- df01 + df02 + beta02
  gamma <- extraDistr::rdirichlet(1, df1)

  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    OMEGA[, , mkeep] <- omega
    GAMMA[mkeep, ] <- gamma
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
      Y1 <- Y1 + y1
      Y2 <- Y2 + y2
    }
    
    if(rp%%disp==0){
      #サンプリング結果を確認
      print(rp)
      print(c(mean(y1), mean(y1_vec)))
      print(c(mean(y2), mean(y2_vec)))
      #print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(phi[, 296:305], phit[, 296:305]), 3))
      print(round(rbind(gamma[296:305], gammat[296:305]), 3))
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

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, 296:305, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[2, 296:305, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[3, 296:305, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[4, 296:305, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(OMEGA[1, 296:305, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(OMEGA[2, 296:305, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(OMEGA[3, 296:305, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(OMEGA[4, 296:305, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")

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





