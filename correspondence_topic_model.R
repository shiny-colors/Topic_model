#####対応トピックモデル#####
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

#set.seed(93441)

####データの生成####
#set.seed(423943)
#文書データの設定
k <- 10   #トピック数
d <- 3000   #文書数
v <- 1000   #語彙数
w <- rpois(d, rgamma(d, 75, 0.5))   #1文書あたりの単語数
a <- 100   #補助変数数
x <- rtpois(d, 22.5, 2, Inf)
f1 <- sum(w)
f2 <- sum(x)

#IDの設定
w_id <- rep(1:d, w)
a_id <- rep(1:d, x)

#パラメータの設定
alpha0 <- rep(0.2, k)   #文書のディレクリ事前分布のパラメータ
alpha1 <- rep(0.15, v)   #単語のディレクリ事前分布のパラメータ
alpha2 <- rep(0.15, a)   #補助情報のディクレリ事前分布のパラメータ

##モデルに基づき単語を生成
for(rp in 1:1000){
  print(rp)
  
  #ディレクリ分布からパラメータを生成
  thetat <- theta <- extraDistr::rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から生成
  phit <- phi <- extraDistr::rdirichlet(k, alpha1)   #単語のトピック分布をディレクリ乱数から生成
  lambda <- matrix(0, nrow=d, ncol=k)   #文書に含むトピックだけを補助情報のトピックにするための確率を格納する行列
  omegat <- omega <- rdirichlet(k, alpha2)   #補助情報のトピック分布をディクレリ乱数から生成
  
  #多項分布の乱数からデータを生成
  WX <- matrix(0, nrow=d, ncol=v)
  AX <- matrix(0, nrow=d, ncol=a)
  word_list <- list()
  aux_list <- list()
  Z1_list <- list()
  Z2_list <- list()
  
  for(i in 1:d){
    #文書のトピックを生成
    z1 <- rmnom(w[i], 1, theta[i, ])   #文書のトピック分布を生成
    z1_vec <- as.numeric(z1 %*% 1:k)
    
    #文書のトピック分布から単語を生成
    word <- rmnom(w[i], 1, phi[z1_vec, ])   #文書のトピックから単語を生成
    word_vec <- colSums(word)   #単語ごとに合計して1行にまとめる
    WX[i, ] <- word_vec  
    
    #文書のトピック分布から補助変数を生成
    #文書で生成させたトピックのみを補助情報のトピック分布とする
    rate <- rep(0, k)
    lambda[i, ] <- colSums(z1) / w[i]

    #補助情報のトピックを生成
    z2 <- rmnom(x[i], 1, lambda[i, ])
    z2_vec <- as.numeric(z2 %*% 1:k)

    #補助情報のトピックから補助情報を生成
    aux <- rmnom(x[i], 1, omega[z2_vec, ])
    aux_vec <- colSums(aux)
    AX[i, ] <- aux_vec
    
    #文書トピックおよび補助情報トピックを格納
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
    word_list[[i]] <- as.numeric(word %*% 1:v)
    aux_list[[i]] <- as.numeric(aux %*% 1:a)
  }
  if(min(colSums(AX)) > 0 & min(colSums(WX)) > 0){
    break
  }
}

#データ行列を整数型行列に変更
Z1 <- do.call(rbind, Z1_list)
Z2 <- do.call(rbind, Z2_list)
wd <- unlist(word_list)
ad <- unlist(aux_list)
storage.mode(WX) <- "integer"
storage.mode(AX) <- "integer"


####トピックモデル推定のためのデータと関数の準備####
##それぞれの文書中の単語の出現および補助情報の出現をベクトルに並べる
##インデックスを作成
doc1_list <- list()
doc1_vec <- list()
word_list <- list()
word_vec <- list()
doc2_list <- list()
doc2_vec <- list()
aux_list <- list()
aux_vec <- list()

for(i in 1:d){
  doc1_list[[i]] <- which(w_id==i)
  doc1_vec[[i]] <- rep(1, length(doc1_list[[i]]))
  doc2_list[[i]] <- which(a_id==i)
  doc2_vec[[i]] <- rep(1, length(doc2_list[[i]]))
}
for(j in 1:v){
  word_list[[j]] <- which(wd==j)
  word_vec[[j]] <- rep(1, length(word_list[[j]]))
}
for(j in 1:a){
  aux_list[[j]] <- which(ad==j)
  aux_vec[[j]] <- rep(1, length(aux_list[[j]]))
}
gc(); gc()


####マルコフ連鎖モンテカルロ法で対応トピックモデルを推定####
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
R <- 5000   #サンプリング回数
keep <- 2   #2回に1回の割合でサンプリング結果を格納
disp <- 10 
burnin <- 1000/keep
iter <- 0

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 1
alpha02 <- 1
beta01 <- 0.1
beta02 <- 0.1

##パラメータの初期値
theta <- rdirichlet(d, rep(1.0, k))  #文書トピックのパラメータの初期値
phi <- rdirichlet(k, rep(0.5, v))   #単語トピックのパラメータの初期値
omega <- rdirichlet(k, rep(0.5, a))   #補助情報トピックのパラメータの初期値

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
OMEGA <- array(0, dim=c(k, a, R/keep))
W_SEG <- matrix(0, nrow=f1, ncol=k)
A_SEG <- matrix(0, nrow=f2, ncol=k)
storage.mode(W_SEG) <- "integer"
storage.mode(A_SEG) <- "integer"
gc(); gc()

##対数尤度の基準値
LLst1 <- sum(WX %*% log(colSums(WX) / f1))
LLst2 <- sum(AX %*% log(colSums(AX) / f2))
LLst <- LLst1 + LLst2


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語トピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_par <- burden_fr(theta, phi, wd, w_id, k)
  word_rate <- word_par$Br

  #多項分布から単語トピックをサンプリング
  Zi1 <- rmnom(f1, 1, word_rate)
  Zi1_T <- t(Zi1)
  z1_vec <- as.numeric(Zi1 %*% 1:k)
  
  
  ##単語トピックのパラメータを更新
  #ディクレリ分布からthetaをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- Zi1_T[, doc1_list[[i]]] %*% doc1_vec[[i]]
  }
  wsum <- wsum0 + alpha01   #ディリクレ分布のパラメータ
  theta <- extraDistr::rdirichlet(d, wsum)   #ディリクレ分布からパラメータをサンプリング
  
  #ディクレリ分布からphiをサンプリング
  vsum0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vsum0[, j] <- Zi1_T[, word_list[[j]], drop=FALSE] %*% word_vec[[j]]
  }
  vsum <- vsum0 + beta01   #ディリクレ分布のパラメータ
  phi <- extraDistr::rdirichlet(k, vsum)   #ディリクレ分布からパラメータをサンプリング

  
  ##補助情報トピックをサンプリング
  #生成させた単語トピックからトピック抽出確率を計算
  theta_aux <- wsum0 / w
  
  #補助情報ごとにトピックの出現率を計算
  aux_par <- burden_fr(theta_aux, omega, ad, a_id, k)
  aux_rate <- aux_par$Br
  
  #多項分布から補助情報トピックをサンプリング
  Zi2 <- rmnom(f2, 1, aux_rate)
  Zi2_T <- t(Zi2)
  z2_vec <- as.numeric(Zi2 %*% 1:k)
  
  ##補助情報トピックのパラメータを更新
  asum0 <- matrix(0, nrow=k, ncol=a)
  for(j in 1:a){
    asum0[, j] <- Zi2_T[, aux_list[[j]], drop=FALSE] %*% aux_vec[[j]]
  }
  asum <- asum0 + beta02   #ディリクレ分布のパラメータ
  omega <- extraDistr::rdirichlet(k, asum)   #ディリクレ分布からパラメータをサンプリング
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    OMEGA[, , mkeep] <- omega
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      W_SEG <- W_SEG + Zi1
      A_SEG <- A_SEG + Zi2
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      LL <- sum(log(rowSums(word_par$Bur))) + sum(log(rowSums(aux_par$Bur)))
      print(rp)
      print(c(LL, LLst))
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(omega[, 1:10], omegat[, 1:10]), 3))
    }
  }
}

####サンプリング結果の可視化と要約####
burnin <- 2000   #バーンイン期間

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA[1, , ]), type="l", ylab="パラメータ", main="文書1のトピック分布のサンプリング結果")
matplot(t(THETA[2, , ]), type="l", ylab="パラメータ", main="文書2のトピック分布のサンプリング結果")
matplot(t(THETA[3, , ]), type="l", ylab="パラメータ", main="文書3のトピック分布のサンプリング結果")
matplot(t(THETA[4, , ]), type="l", ylab="パラメータ", main="文書4のトピック分布のサンプリング結果")
matplot(t(THETA[5, , ]), type="l", ylab="パラメータ", main="文書4のトピック分布のサンプリング結果")

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, , ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[3, , ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[5, , ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[7, , ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(PHI[9, , ]), type="l", ylab="パラメータ", main="トピック5の単語の出現率のサンプリング結果")

#補助情報の出現確率のサンプリング結果
matplot(t(OMEGA[2, , ]), type="l", ylab="パラメータ", main="トピック1の補助情報の出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[4, , ]), type="l", ylab="パラメータ", main="トピック2の補助情報の出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[6, , ]), type="l", ylab="パラメータ", main="トピック3の補助情報の出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[8, , ]), type="l", ylab="パラメータ", main="トピック4の補助情報の出現率のパラメータのサンプリング結果")
matplot(t(OMEGA[10, , ]), type="l", ylab="パラメータ", main="トピック5の補助情報の出現率のパラメータのサンプリング結果")

##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(cbind(t(word_mu), t(phit)), 3)

#補助情報出現率の事後推定量
tag_mu1 <- apply(OMEGA[, , burnin:(R/keep)], c(1, 2), mean)   #補助情報の出現率の事後平均
round(cbind(t(tag_mu1), t(omegat)), 3)

