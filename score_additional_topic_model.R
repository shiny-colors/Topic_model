#####教師ありトピックモデル####
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


####データの発生####
#set.seed(423943)
#データの設定
s <- 5   #教師数
k1 <- 3   #教師ごとのトピック数
k2 <- 10   #文書集合全体のトピック数
d <- 2500   #文書数
v1 <- 300   #教師に関係のある語彙数
v2 <- 100   #教師に関係のない語彙数
v <- v1 + v2   #語彙数
w <- rpois(d, rgamma(d, 55, 0.50))   #1文書あたりの単語数
f <- sum(w)

#インデックスを作成
id_vec <- rep(1:d, w)
index_hh <- list()
for(i in 1:d){
  index_hh[[i]] <- which(id_vec==i)
}

#教師データを生成
pr <- runif(s, 1, 5)
Y <- rmnom(d, 1, pr)
y <- as.numeric(Y %*% 1:s)
y_vec <- as.numeric(y)[id_vec] 
index_y <- list()
for(j in 1:s) {index_y[[j]] <- which(y==j)}
y_freq <- as.numeric(table(y))


##パラメータの設定
#教師ごとのトピックを生成
alpha0 <- rep(0.3, k1)   #教師データの文書のディレクリ事前分布のパラメータ
alpha1 <- list()
for(j in 1:s){
  alpha1[[j]] <- c(rep(0.4, v1), rep(0.005, v2))   #教師に関係のある単語のディレクリ事前分布のパラメータ
}
alpha2 <- c(rep(0.1, v1), rep(5, v2))   #教師に関係のない単語のディレクリ事前分布のパラメータ

#ディレクリ乱数の発生
thetat <- theta <- rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から発生
phit <- phi <- list()
for(j in 1:s) {phit[[j]] <- phi[[j]] <- rdirichlet(k1, alpha1[[j]])}   #評点に関係のある単語分布をディレクリ乱数から発生
gammat <- gamma <- rdirichlet(1, alpha2)   #評点に関係のない単語分布をディレクリ乱数から発生
betat <- beta <- rbeta(sum(f), 15, 15)   #単語が教師と関連するかどうかのパラメータ


##多項分布の乱数からデータを発生
WX <- matrix(0, nrow=d, ncol=v)
x_list <- list()
x <- rep(0, f)
Z <- list()
index_v1 <- 1:v1
index_v2 <- (v1+1):v

for(i in 1:d){
  
  #文書のトピックを生成
  z <- rmnom(w[i], 1, theta[i, ])   #文書のトピック分布を発生
  z_vec <- z %*% c(1:k1)   #トピック割当をベクトル化
  
  #一般語かどうかを生成
  x_list[[i]] <- rbinom(w[i], 1, beta[index_hh[[i]]])
  
  phi[[y[i]]][z_vec[x_list[[i]]==1], ]
  phi
  #生成したトピックから単語を生成
  wn <- rmnom(sum(x_list[[i]]), 1, phi[[y[i]]][z_vec[x_list[[i]]==1], ])   #文書のトピックから単語を生成
  an <- rmnom(sum(1-x_list[[i]]), 1, gammat)
  wdn <- colSums(wn) + colSums(an)   #単語ごとに合計して1行にまとめる
  WX[i, ] <- wdn
  Z[[i]] <- z
  x[index_hh[[i]]] <- x_list[[i]]
  print(i)
}

####トピックモデル推定のためのデータと関数の準備####
##それぞれの文書中の単語の出現および補助情報の出現をベクトルに並べる
##データ推定用IDを作成
ID_list <- list()
wd_list <- list()

#文書ごとに文書IDおよび単語IDを作成
for(i in 1:nrow(WX)){
  print(i)
  
  #単語のIDベクトルを作成
  ID_list[[i]] <- rep(i, w[i])
  num1 <- (WX[i, ] > 0) * (1:v)
  num2 <- which(num1 > 0)
  W1 <- WX[i, (WX[i, ] > 0)]
  number <- rep(num2, W1)
  wd_list[[i]] <- number
}

#リストをベクトルに変換
ID_d <- unlist(ID_list)
wd <- unlist(wd_list)

##インデックスを作成
doc_list <- list()
word_list <- list()
y_list <- list()
index_s <- list()
words_list <- list()
docs_list <- list()
docs_vec <- list()
wd0 <- list()
w0 <- list()

for(i in 1:length(unique(ID_d))) {doc_list[[i]] <- which(ID_d==i)}
for(i in 1:length(unique(wd))) {word_list[[i]] <- which(wd==i)}
for(j in 1:s) {y_list[[j]] <- which(y==j)}
for(j in 1:s) {index_s[[j]] <- which(y_vec==j)}

for(i in 1:s){
  words_list[[i]] <- list()
  wds <- wd[index_s[[i]]]
  
  for(j in 1:length(unique(wd))){
    words_list[[i]][[j]] <- which(wds==j)
  }
}

for(j in 1:s){
  wd0[[j]] <- wd[index_s[[j]]]
  w0[[j]] <- w[index_y[[j]]]
  docs_list[[j]] <- list()
  dcs <- ID_d[index_s[[j]]]
  vec <- unique(dcs)
  for(i in 1:length(vec)){
    docs_list[[j]][[i]] <- which(dcs==vec[i])
    docs_vec[[j]] <- vec
  }
}


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
burnin <- 1000/keep

##事前分布の設定
#ハイパーパラメータの事前分布
alpha01 <- 1.0
alpha02 <- 0.5


##パラメータの初期値
#トピック分布の初期値
theta <- extraDistr::rdirichlet(d, rep(1, k1))   #文書トピックのパラメータの初期値


#一般語の単語分布の初期値
inv_idf <- colSums(WX > 0)/d
gamma <- inv_idf / sum(inv_idf)

#教師関連単語分布の初期値
phi <- list()
for(j in 1:s) {
  M <- colSums(WX[y==j, ])
  phi[[j]] <- extraDistr::rdirichlet(k1, (M+1)*log(1/inv_idf))
}


#混合率の初期値
rd <- matrix(0, nrow=f, ncol=2)
rd[, 1] <- 0.5
rd[, 2] <- 0.5


##パラメータの格納用配列
THETA <- array(0, dim=c(d, k1, R/keep))
PHI1 <- PHI2 <- PHI3 <- PHI4 <- PHI5 <- array(0, dim=c(k1, v, R/keep))
GAMMA <- matrix(0, nrow=R/keep, ncol=v)
SEG1 <- matrix(0, nrow=f, ncol=k1)
SEG2 <- rep(0, nrow=f)
storage.mode(SEG1) <- "integer"
gc(); gc()

##MCMC推定用配列
wsum0 <- matrix(0, nrow=d, ncol=k1)
vf0 <- matrix(0, nrow=k1, ncol=v)
zf0 <- rep(0, v)
switching_rate <- rep(0, f)
Zi1_full <- matrix(0, nrow=f, ncol=k1)
Zi2_zeros <- rep(0, f)
vec1 <- 1/1:k1


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  #パラメータの格納用配列を更新
  Brs <- list()
  Zi1 <- list()
  Zi2 <- list()
  
  for(j in 1:s){
    ##教師単位でのトピック分布をサンプリング
    #単語ごとにトピック分布のパラメータを推定
    theta0 <- theta[index_y[[j]], ]
    Brs[[j]] <- burden_fr(theta0, phi[[j]], wd0[[j]], w0[[j]], k1)
    word_rate <- Brs[[j]]$Br
    
    #多項分布よりトピックをサンプリング
    n <- nrow(word_rate)
    rand1 <- matrix(runif(n), nrow=n, ncol=k1)
    user_cumsums <- rowCumsums(word_rate)
    Zi1[[j]] <- ((k1+1) - (user_cumsums > rand1) %*% rep(1, k1)) %*% vec1   #トピックをサンプリング
    Zi1[[j]][Zi1[[j]]!=1] <- 0
    Zi1_full[index_s[[j]], ] <- Zi1[[j]]
    
    
    ##単語ごとに教師トピックと関係があるかどうかをサンプリング
    #二項分布のパラメータを推定
    LLz1 <- rd[index_s[[j]], 1] * rowSums(Brs[[j]]$Bur)
    LLz2 <- rd[index_s[[j]], 2] * gamma[wd0[[j]]]
    switching_rate[index_s[[j]]] <- LLz1 / (LLz1+LLz2)
    
    #二項分布から関係有無を生成
    Zi2[[j]] <- rbinom(length(index_s[[j]]), 1, switching_rate[index_s[[j]]])
  }
  
  ##教師ごとにパラメータをサンプリング
  for(j in 1:s){
    #文書のトピックのパラメータを推定
    zi <- Zi1[[j]] * matrix(Zi2[[j]], nrow=length(index_s[[j]]), ncol=k1)
    Zi2_zeros[index_s[[j]]] <- 1-Zi2[[j]]
    
    for(i in 1:y_freq[j]){
      wsum0[index_y[[j]][i], ] <- colSums(zi[docs_list[[j]][[i]], ])
    }
    
    #単語分布のパラメータを推定
    for(l in 1:v){
      vf0[, l] <- colSums(zi[words_list[[j]][[l]], ,drop=FALSE])
    }
    #ディクレリ分布から単語分布をサンプリング
    vf <- vf0 + alpha02
    phi[[j]] <- extraDistr::rdirichlet(k1, vf)
  }
  
  #ディクレリ分布からトピック分布をサンプリング
  wsum <- wsum0 + alpha01
  theta <- extraDistr::rdirichlet(d, wsum)
  
  
  ##一般語のパラメータをサンプリング
  #一般語パラメータgammaをサンプリング
  for(j in 1:v){
    zf0[j] <- sum(Zi2_zeros[word_list[[j]]])
  }
  zf <- zf0 + alpha02
  gamma <- extraDistr::rdirichlet(1, zf)
  
  #混合率をサンプリング
  Zi2_ones <- 1-Zi2_zeros
  for(i in 1:d){
    rd[i, 1] <- mean(Zi2_ones[doc_list[[i]]])
  }
  rd[, 2] <- 1-rd[, 1]
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA[, , mkeep] <- theta
    PHI1[, , mkeep] <- phi[[1]]
    PHI1[, , mkeep] <- phi[[2]]
    PHI1[, , mkeep] <- phi[[3]]
    PHI1[, , mkeep] <- phi[[4]]
    PHI1[, , mkeep] <- phi[[5]]
    GAMMA[mkeep, ] <- gamma
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      SEG1 <- SEG1 + Zi1_full      
      SEG2 <- SEG2 + Zi2_ones
    }
    
    #サンプリング結果を確認
    print(rp)
    print(round(cbind(theta[y==1, ][1:7, ], thetat[y==1, ][1:7, ], theta[y==2, ][1:7, ], thetat[y==2, ][1:7, ],
                      theta[y==3, ][1:7, ], thetat[y==3, ][1:7, ]), 3))
    print(round(cbind(phi[[1]][, 296:305], phit[[1]][, 296:305]), 3))
    print(round(cbind(phi[[2]][, 296:305], phit[[2]][, 296:305]), 3))
    print(round(cbind(phi[[3]][, 296:305], phit[[3]][, 296:305]), 3))
    print(round(rbind(gamma[290:309], gammat[290:309]), 3))
  }
}



####サンプリング結果の可視化と要約####
burnin <- 1000/keep   #バーンイン期間
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
