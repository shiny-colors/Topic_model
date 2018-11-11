#####Latent Dirichlet Allocationモデル(高速化)#####
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
#set.seed(423943)
#データの設定
k <- 15   #トピック数
d <- 3000   #文書数
v <- 1000   #語彙数
w <- rpois(d, rgamma(d, 60, 0.50))   #1文書あたりの単語数
f <- sum(w)
vec <- rep(1, k)

#IDの設定
d_id <- rep(1:d, w)

#パラメータの設定
alpha01 <- rep(0.15, k)   #文書のディレクリ事前分布のパラメータ
alpha02 <- rep(0.1, v)   #単語のディレクリ事前分布のパラメータ

##すべての単語が生成されるまで繰り返す
for(rp in 1:1000) {
  print(rp)
  
  #ディレクリ乱数の発生
  thetat <- theta <- rdirichlet(d, alpha01)   #文書のトピック分布をディレクリ乱数から発生
  phit <- phi <- rdirichlet(k, alpha02)   #単語のトピック分布をディレクリ乱数から発生
  
  #多項分布の乱数からデータを発生
  WX <- matrix(0, nrow=d, ncol=v)
  wd_list <- Z<- list() 
  
  for(i in 1:d){
    #トピックを生成
    z <- rmnom(w[i], 1, theta[i, ])   #文書のトピック分布を発生
    z_vec <- z %*% c(1:k)   #トピック割当をベクトル化
    
    #単語を生成
    wx <- rmnom(w[i], 1, phi[z_vec, ])   #文書のトピックカラ単語を生成
    wd_list[[i]] <- as.numeric(wx %*% 1:v)   #単語ベクトルを格納
    WX[i, ] <- colSums(wx)   #単語ごとに合計して1行にまとめる
    Z[[i]] <- z
  }
  if(min(colSums(WX)) > 0) break
}

#リストをベクトルに変換
wd <- unlist(wd_list)
sparse_data <- sparseMatrix(i=1:f, wd, x=rep(1, f), dims=c(f, v))
sparse_data_T <- t(sparse_data)


####マルコフ連鎖モンテカルロ法で対応トピックモデルを推定####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, d_id, k, vec){
  
  #負担係数を計算
  Bur <- theta[d_id, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / as.numeric(Bur %*% vec)   #負担率の計算
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}

##アルゴリズムの設定
R <- 2000   #サンプリング回数
keep <- 2   #2回に1回の割合でサンプリング結果を格納
disp <- 20
iter <- 0
burnin <- 200/keep

##データの設定
d_vec <- sparseMatrix(sort(d_id), 1:f, x=rep(1, f), dims=c(d, f))


##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 0.1
alpha02 <- 0.1


##パラメータの初期値
theta <- extraDistr::rdirichlet(d, rep(1, k))   #文書トピックのパラメータの初期値
phi <- extraDistr::rdirichlet(k, rep(1, v))    #単語トピックのパラメータの初期値

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG) <- "integer"
gc(); gc()


##対数尤度の基準値
LLst <- sum(sparse_data %*% log(colSums(WX) / sum(WX)))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語トピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_par <- burden_fr(theta, phi, wd, d_id, k, vec)
  word_rate <- word_par$Br
  
  #多項分布から単語トピックをサンプリング
  Zi <- rmnom(f, 1, word_rate)   
  
  
  ##単語トピックのパラメータを更新
  #ディクレリ分布からthetaをサンプリング
  wsum <- d_vec %*% Zi + alpha01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #ディクレリ分布からphiをサンプリング
  vsum <- t(sparse_data_T %*% Zi) + alpha02
  phi <- extraDistr::rdirichlet(k, vsum)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #パラメータを格納
  if(rp%%keep==0){
    #モデルのパラメータを格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #バーンイン期間を超えたらトピックを格納
    if(rp >= burnin){
      SEG <- SEG + Zi
    }
  }
  
  #対数尤度とサンプリング結果を確認
  if(rp%%disp==0){
    #トピックモデルの対数尤度
    LL <- sum(log((theta[d_id, ] * t(phi)[wd, ]) %*% vec))
    
    #サンプリング結果を表示
    print(rp)
    print(c(LL, LLst))
    print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 200/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#文書のトピック分布のサンプリング結果
matplot(t(THETA[1, , ]), type="l", ylab="パラメータ", main="文書1のトピック分布のサンプリング結果")
matplot(t(THETA[2, , ]), type="l", ylab="パラメータ", main="文書2のトピック分布のサンプリング結果")
matplot(t(THETA[3, , ]), type="l", ylab="パラメータ", main="文書3のトピック分布のサンプリング結果")
matplot(t(THETA[4, , ]), type="l", ylab="パラメータ", main="文書4のトピック分布のサンプリング結果")

#単語の出現確率のサンプリング結果
matplot(t(PHI[1, 1:10, ]), type="l", ylab="パラメータ", main="トピック1の単語の出現率のサンプリング結果")
matplot(t(PHI[2, 11:20, ]), type="l", ylab="パラメータ", main="トピック2の単語の出現率のサンプリング結果")
matplot(t(PHI[3, 21:30, ]), type="l", ylab="パラメータ", main="トピック3の単語の出現率のサンプリング結果")
matplot(t(PHI[4, 31:40, ]), type="l", ylab="パラメータ", main="トピック4の単語の出現率のサンプリング結果")
matplot(t(PHI[5, 41:50, ]), type="l", ylab="パラメータ", main="トピック5の単語の出現率のサンプリング結果")
matplot(t(PHI[6, 51:60, ]), type="l", ylab="パラメータ", main="トピック6の単語の出現率のサンプリング結果")
matplot(t(PHI[7, 61:70, ]), type="l", ylab="パラメータ", main="トピック7の単語の出現率のサンプリング結果")
matplot(t(PHI[8, 71:80, ]), type="l", ylab="パラメータ", main="トピック8の単語の出現率のサンプリング結果")

##サンプリング結果の要約推定量
#トピック分布の事後推定量
topic_mu <- apply(THETA[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu, thetat), 3)
round(topic_sd <- apply(THETA[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語出現確率の事後推定量
word_mu <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(rbind(word_mu, phit)[, 1:50], 3)
