#####Hidden Topic Marcov Model#####
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
k <- 10   #トピック数
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
w_id <- rep(1:d, words)
x_vec <- rep(0, f)
x_vec[c(1, cumsum(w[-a])+1)] <- 1

##パラメータの設定
#ディレクリ分布のパラメータ
alpha01 <- rep(0.15, k)
alpha11 <- rep(0.1, v) 

#パラメータを生成
for(rp in 1:100){
  theta0 <- thetat0 <- extraDistr::rdirichlet(d, alpha01)
  phi <- phit <- extraDistr::rdirichlet(k, alpha11)
  beta0 <- beta0t <- -2.75
  beta1 <- beta1t <- 2.0
  
  ##モデルにもとづき単語を生成する
  wd_list <- list()
  ID_list <- list()
  td_list <- list()
  Z1_list <- list()
  Z2_list <- list()
  
  for(i in 1:d){
    if(i%%100==0){
      print(i)
    }
    
    freq <- words[i]
    index_w <- which(w_id==i)
    z1_vec <- rep(0, freq)
    z2_vec <- rep(0, freq)
      
    for(j in 1:freq){
      if(j==1){
        #文書の開始トピックを生成
        z2 <- rmnom(1, 1, theta0[i, ])
        z2_vec[j] <- as.numeric(z2 %*% 1:k)
        
      } else {
        
        #トピックの切換えを生成
        z2_vec[j-1]
        logit <- beta0 + beta1 * x_vec[index_w[j]]
        pr <- exp(logit)/(1+exp(logit))
        z1_vec[j] <- rbinom(1, 1, pr)
        
        #文書の2単語目以降のトピックを生成
        if(z1_vec[j]==1){
          #トピック切換えがあった場合新たなトピックを生成
          z2 <- rmnom(1, 1, theta0[i, ])
          z2_vec[j] <- as.numeric(z2 %*% 1:k)
          
        } else {
          
          #トピック切換えがなかった場合1つ前と同じトピックを生成
          z2_vec[j] <- z2_vec[j-1]
        }
      }
    }
    #トピックにもとづき単語を生成
    wn <- rmnom(freq, 1, phi[z2_vec, ])
    wd_list[[i]] <- as.numeric(wn %*% 1:v)
    
    #パラメータを格納
    ID_list[[i]] <- rep(i, freq)
    td_list[[i]] <- 1:freq
    Z1_list[[i]] <- z1_vec
    Z2_list[[i]] <- z2_vec
  }
  
  #リストをベクトル変換
  ID_d <- unlist(ID_list)
  td_d <- unlist(td_list)
  wd <- unlist(wd_list)
  z1 <- unlist(Z1_list)
  z2 <- unlist(Z2_list)
  WX <- sparseMatrix(i=1:f, j=wd, x=rep(1, f), dims=c(f, v))   #スパース行列を作成
  theta <- thetat <- matrix(as.numeric(table(ID_d, z2) / words), nrow=d, ncol=k)
  if(length(which(colSums(WX)==0))==0) break
}
sum(colSums(WX)==0)

##インデックスを作成
doc_list <- list()
word_list <- list()
for(i in 1:d){doc_list[[i]] <- which(ID_d==i)}
for(i in 1:v){word_list[[i]] <- which(wd==i)}


####マルコフ連鎖モンテカルロ法でHTMモデルを推定####
##ロジスティック回帰モデルの対数尤度関数
#ロジスティック回帰モデルの対数尤度を定義
loglike <- function(alpha, beta, x, y){
  
  #尤度を定義して合計する
  logit <- alpha + x * beta 
  p <- exp(logit) / (1 + exp(logit))
  LLS <- y*log(p) + (1-y)*log(1-p)  
  LL <- sum(LLS)
  return(LL)
}

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


####LDAで初期値を生成####
##アルゴリズムの設定
R0 <- 1000
keep0 <- 2  
iter0 <- 0
burnin0 <- 200/keep0
disp0 <- 10

#単語トピック単位のパラメータの初期値
word_data <- matrix(0, nrow=d, ncol=v)
for(i in 1:d){
  word_data[i, ] <- colSums(WX[doc_list[[i]], ])
}
tf0 <- colSums(word_data)/sum(word_data)
idf0 <- log(d / colSums(word_data > 0))
theta <- extraDistr::rdirichlet(d, rep(0.3, k))   #文書単位のトピックの初期値
phi0 <- extraDistr::rdirichlet(k, tf0*10) + 0.001   #文書単位の出現確率の初期値
phi <- phi0 / rowSums(phi0)

#ハイパーパラメータの設定
alpha01 <- 1
beta0 <- 0.5

#パラメータの格納用配列
THETA0 <- array(0, dim=c(d, k, R/keep))
PHI0 <- array(0, dim=c(k, v, R/keep))
SEG0 <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG0) <- "integer"

#対数尤度の基準値
LLst <- sum(WX %*% log(colSums(WX)/sum(WX)))


#ギブスサンプリングでパラメータをサンプリング
for(rp in 1:R0){
  
  ##単語トピックをサンプリング
  #単語ごとにトピックの出現率を計算
  word_par <- burden_fr(theta, phi, wd, words, k)
  word_rate <- word_par$Br
  
  #多項分布から単語トピックをサンプリング
  Zi <- rmnom(f, 1, word_rate)   
  z_vec <- Zi %*% 1:k
  
  ##単語トピックのパラメータを更新
  #ディクレリ分布からthetaをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi[doc_list[[i]], ])
  }
  wsum <- wsum0 + alpha01 
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #ディクレリ分布からphiをサンプリング
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi[word_list[[j]], , drop=FALSE])
  }
  vf <- vf0 + beta0
  phi <- extraDistr::rdirichlet(k, vf)
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep0==0){
    #サンプリング結果の格納
    mkeep <- rp/keep0
    THETA0[, , mkeep] <- theta
    PHI0[, , mkeep] <- phi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep0==0 & rp >= burnin0){
      SEG0 <- SEG0 + Zi
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      print(c(sum(log(rowSums(word_par$Bur))), LLst))
      print(round(cbind(theta[1:10, ], thetat[1:10, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
    }
  }
}


####HTMモデルのMCMCアルゴリズムの設定####
##アルゴリズムの設定
R <- 10000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##パラメータの真値
theta <- thetat
phi <- phit
beta0 <- beta0t
beta1 <- beta1t
r0 <- exp(beta0)/(1+exp(beta0))
r1 <- exp(beta0+beta1)/(1+exp(beta0+beta1))
r <- cbind(r0, r1)

##LDAからHTMモデルの初期値を設定
theta <- apply(THETA0[, , burnin0:(R0/keep0)], c(1, 2), mean)
phi <- apply(PHI0[, , burnin0:(R0/keep0)], c(1, 2), mean)
beta0 <- -2.5
beta1 <- 1.5
r0 <- exp(beta0)/(1+exp(beta0))
r1 <- exp(beta0+beta1)/(1+exp(beta0+beta1))
r <- cbind(r0, r1)

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 0.01 
beta01 <- 1
betas <- rep(0, 2)  #回帰係数の初期値
B0 <- 0.01*diag(2)
rw <- 0.025

##パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
BETA <- matrix(0, nrow=R/keep, 2)
SEG1 <- rep(0, f)
SEG2 <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG1) <- "integer"
storage.mode(SEG2) <- "integer"


##MCMC推定用配列
max_word <- max(words)
index_t11 <- which(td_d==1)
index_t21 <- list()
index_t22 <- list()
for(j in 2:max_word){
  index_t21[[j]] <- which(td_d==j)-1
  index_t22[[j]] <- which(td_d==j)
}


####ギブスサンプリングでHTMモデルのパラメータをサンプリング####
for(rp in 1:R){
  
  #単語ごとのトピック尤度とトピックの候補を生成
  word_par <- burden_fr(theta, phi, wd, words, k)   
  Li01 <- word_par$Bur   #トピックモデルの期待尤度
  Zi02 <- rmnom(f, 1, word_par$Br)     #多項分布からトピックを生成
  z02 <- as.numeric(Zi02 %*% 1:k)

  #マルコフ推移モデルの尤度と割当確率を逐次的に推定
  Zi1 <- rep(0, f)
  z1_rate <- rep(0, f)
  rf02 <- rep(0, f)
  
  for(j in 2:max_word){
    
    #データの設定
    index <- index_t22[[j]]
    x0 <- x_vec[index]
    Li01_obz <- Li01[index, , drop=FALSE]
    Zi02_obz <-  Zi02[index_t21[[j]], ]
    
    #マルコフ切換え確率を推定
    Li11 <- (1-r[x0+1]) * rowSums(Li01_obz * Zi02_obz)
    Li12 <- r[x0+1] * rowSums(Li01_obz * (1-Zi02_obz))
    z1_rate[index] <- Li12 / (Li11+Li12)

    #ベルヌーイ分布から切換え変数を生成
    Zi1[index] <- rbinom(length(index), 1, z1_rate[index])
    index_z1 <- which(Zi1[index]==0)
    if(length(index_z1)==0) next
    
    if(length(index)!=1){
      Zi02[index, ][index_z1, ] <- Zi02[index_t21[[j]], , drop=FALSE][index_z1, ]
    } else {
      Zi02[index, ] <- Zi02[index_t21[[j]], ]
    }
  }
  Zi2 <- Zi02
  z2_vec <- as.numeric(Zi2 %*% 1:k)


  ##MH法で混合率の回帰パラメータをサンプリング
  #データの設定
  y <- Zi1[-index_t11]
  x <- x_vec[-index_t11]

  #betaのサンプリング
  betad <- c(beta0, beta1)
  betan <- betad + rw * rnorm(2)   #新しいbetaをランダムウォークでサンプリング
  
  #対数尤度と対数事前分布の計算
  lognew <- loglike(betan[1], betan[2], x, y)
  logold <- loglike(betad[1], betad[2], x, y)
  logpnew <- lndMvn(betan, betas, B0)
  logpold <- lndMvn(betad, betas, B0)
  
  
  #MHサンプリング
  alpha <- min(1, exp(lognew + logpnew - logold - logpold))
  if(alpha == "NAN") alpha <- -1
  
  #一様乱数を発生
  u <- runif(1)
  
  #u < alphaなら新しいbetaを採択
  if(u < alpha){
    beta0 <- betan[1]
    beta1 <- betan[2]
    
    #そうでないならbetaを更新しない
  } else {
    beta0 <- betad[1]
    beta1 <- betad[2]
  }
  
  #切換え確率の混合率を更新
  r0 <- exp(beta0) / (1+exp(beta0))
  r1 <- exp(beta0+beta1) / (1+exp(beta0+beta1))
  r <- cbind(r0, r1)

  ##トピックモデルのパラメータをサンプリング
  #トピック分布thetaをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- colSums(Zi2[doc_list[[i]], ])
  }
  wsum <- wsum0 + beta01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  #単語分布phiをサンプリング
  vf0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vf0[, j] <- colSums(Zi2[word_list[[j]], , drop=FALSE])
  }
  vf <- vf0 + alpha11
  phi <- extraDistr::rdirichlet(k, vf)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    BETA[mkeep, ] <- c(beta0, beta1)
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(mkeep >= burnin & rp%%keep==0){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      print(rp)
      print(c(sum(log(rowSums(word_par$Bur))), LLst))
      print(round(cbind(theta[1:6, ], thetat[1:6, ]), 3))
      print(round(cbind(phi[, 1:10], phit[, 1:10]), 3))
      round(print(c(beta0, beta1, r)), 3)
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





