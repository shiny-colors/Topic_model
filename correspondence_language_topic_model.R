#####対応言語トピックモデル#####
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

####データの発生####
##文書の設定
k1 <- 10   #名詞のトピック数
k2 <- 15   #動詞と形容詞のトピック数
d <- 2000   #文書数
v1 <- 1000   #名詞の語彙数
v2 <- 500   #動詞(形容詞)の語彙数
s0 <- rpois(d, rgamma(d, 22.5, 0.5))   #動詞と名詞のペア数
s1 <- extraDistr::rtpois(sum(s0), 4.0, a=0, b=Inf)   #ペアごとの名詞数
s2 <- extraDistr::rtpois(sum(s0), 0.7, a=0, b=3)   #ペアごとの動詞数
f0 <- sum(s0)   #総文章数
f1 <- sum(s1)   #名詞の総単語数
f2 <- sum(s2)   #動詞の総単語数


##IDを設定
p_id <- rep(1:d, s0)
w <- as.numeric(tapply(s1, p_id, sum))   #単語数
w_id <- rep(1:d, w)
w_id1 <- rep(1:f0, s1)
w_id2 <- rep(1:f0, s2)


##モデルに基づき単語を生成
#ディレクリ分布のパラメータ
alpha01 <- rep(0.1, k1)   #トピック分布のパラメータ
alpha02 <- rep(0.05, k2)   #混合モデルのパラメータ
beta01 <- rep(0.1, v1)   #名詞の単語分布のパラメータ
beta02 <- rep(0.1, v2)   #動詞の単語分布のパラメータ

#パラメータを生成
theta1 <- thetat1 <- extraDistr::rdirichlet(d, alpha01)
beta <- betat <- matrix(rgamma(k1*k2, 0.25, 0.5), nrow=k1, ncol=k2)
phi <- phit <- extraDistr::rdirichlet(k1, beta01)
psi <- psit <- extraDistr::rdirichlet(k2, beta02)

#逐次的にトピックと単語を生成
WX1 <- matrix(0, nrow=f0, ncol=v1)
WX2 <- matrix(0, nrow=f0, ncol=v2)
word_list1 <- list()
word_list2 <- list()
Z_list1 <- list()
Z_list2 <- list()
Z_sums <- matrix(0, nrow=k2, ncol=k1)
Pr <- matrix(0, nrow=f0, ncol=k2)

for(i in 1:f0){
  if(i%%1000==0){
    print(i)
  }
  #名詞トピックを生成
  pr1 <- theta1[p_id[i], ]
  z1 <- rmnom(s1[i], 1, pr1)
  z1_vec <- as.numeric(z1 %*% 1:k1)

  #動詞トピックを生成
  U <- exp(colSums(z1) %*% beta)
  pr2 <- U / sum(U)
  z2 <- rmnom(s2[i], 1, pr2)
  z2_vec <- as.numeric(z2 %*% 1:k2)

  #トピックにもとづき単語を生成
  word1 <- rmnom(s1[i], 1, phi[z1_vec, ])
  word2 <- rmnom(s2[i], 1, psi[z2_vec, ])

  #データを格納
  Z_list1[[i]] <- z1
  Z_list2[[i]] <- z2
  WX1[i, ] <- colSums(word1)
  WX2[i, ] <- colSums(word2)
  word_list1[[i]] <- as.numeric(word1 %*% 1:v1)
  word_list2[[i]] <- as.numeric(word2 %*% 1:v2)
  
  #theta2を推定のためにトピックを格納
  for(j in 1:s2[i]){
    Z_sums[z2_vec[j], ] <- Z_sums[z2_vec[j], ] + colSums(z1)
  }
  Pr[i, ] <- pr2
}

#リストを変換
Z1 <- do.call(rbind, Z_list1)
Z2 <- do.call(rbind, Z_list2)
word_vec1 <- unlist(word_list1)
word_vec2 <- unlist(word_list2)

#動詞のトピック分布を生成
theta2 <- thetat2 <- extraDistr::rdirichlet(k2, Z_sums + 0.01)


####マルコフ連鎖モンテカルロ法で対応言語トピックモデルを推定####
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
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##インデックスを作成
doc_list <- list()
doc_vec <- list()
w_list1 <- list()
w_vec1 <- list()
w_list2 <- list()
w_vec2 <- list()
pair_list <- list()
pair_vec <- list()

for(i in 1:d){
  doc_list[[i]] <- which(w_id==i)
  doc_vec[[i]] <- rep(1, length(doc_list[[i]]))
}
for(j in 1:v1){
  w_list1[[j]] <- which(word_vec1==j)
  w_vec1[[j]] <- rep(1, length(w_list1[[j]]))
}
for(j in 1:v2){
  w_list2[[j]] <- which(word_vec2==j)
  w_vec2[[j]] <- rep(1, length(w_list2[[j]]))
}
for(i in 1:f0){
  if(i%%1000==0){print(i)}
  pair_list[[i]] <- which(w_id1==i)
  pair_vec[[i]] <- rep(1, length(pair_list[[i]]))
}

##ペアIDを作成
pair_id_list1 <- list()
for(i in 1:f0){
  pair_id_list1[[i]] <- rep(pair_list[[i]], s2[i])
}
pair_id1 <- unlist(pair_id_list1)
pair_id2 <- rep(1:f2, rep(s1, s2))


##パラメータの事前分布
alpha01 <- 1
alpha02 <- 1
beta01 <- 0.1
beta02 <- 0.1

##パラメータの真値
theta1 <- thetat1
theta2 <- thetat2
phi <- phit
psi <- psit

##パラメータの初期値
theta1 <- extraDistr::rdirichlet(d, rep(1, k1))
theta2 <- extraDistr::rdirichlet(k2, rep(1, k1))
phi <- extraDistr::rdirichlet(k1, rep(1, v1))
psi <- extraDistr::rdirichlet(k2, rep(1, v2))

##パラメータの格納用配列
THETA1 <- array(0, dim=c(d, k1, R/keep))
THETA2 <- array(0, dim=c(k1, k2, R/keep))
PHI <- array(0, dim=c(k1, v1, R/keep))
PSI <- array(0, dim=c(k2, v2, R/keep))
SEG1 <- matrix(0, nrow=f1, ncol=k1)
SEG2 <- matrix(0, nrow=f2, ncol=k2)
LLho <- rep(0, R/keep)

##対数尤度の基準値
LLst1 <- sum(WX1 %*% log((colSums(WX1)+0.1) / (sum(WX1)+0.1*v1)))
LLst2 <- sum(WX2 %*% log((colSums(WX2)+0.1) / (sum(WX2)+0.1*v2)))
LLst <- LLst1 + LLst2


####マルコフ連鎖モンテカルロ法でパラメータをサンプリング####
for(rp in 1:R){
  
  ##名詞トピックをサンプリング
  #潜在トピックのパラメータを推定
  par1 <- burden_fr(theta1, phi, word_vec1, w_id, k1)
  z1_rate <- par1$Br
  
  #多項分布よりトピックをサンプリング
  Zi1 <- rmnom(f1, 1, z1_rate)
  z1_vec <- as.numeric(Zi1 %*% 1:k1)
  Zi1_T <- t(Zi1)
  
  
  ##パラメータを更新
  #トピック分布のパラメータを更新
  wsum0 <- matrix(0, nrow=d, ncol=k1)
  for(i in 1:d){
    wsum0[i, ] <- Zi1_T[, doc_list[[i]]] %*% doc_vec[[i]]
  }
  wsum <- wsum0 + alpha01   #ディリクレ分布のパラメータ
  theta1 <- extraDistr::rdirichlet(d, wsum)   #ディリクレ分布からパラメータをサンプリング
  
  #単語分布のパラメータを更新
  vsum0 <- matrix(0, nrow=k1, ncol=v1)
  for(j in 1:v1){
    vsum0[, j] <- Zi1_T[, w_list1[[j]], drop=FALSE] %*% w_vec1[[j]]
  }
  vsum <- vsum0 + beta01   #ディリクレ分布のパラメータ
  phi <- extraDistr::rdirichlet(k1, vsum)   #ディリクレ分布からパラメータをサンプリング
  
  
  ##動詞のトピックをサンプリング
  #生成した名詞トピックからトピック分布を推定
  topic_par <- matrix(0, nrow=f0, ncol=k2)
  par2 <- t(t(log(theta2))[z1_vec, ])
  for(i in 1:f0){
    topic_par[i, ] <- par2[, pair_list[[i]], drop=FALSE] %*% pair_vec[[i]]   #トピック分布の対数尤度
  }
  
  word_par <- t(log(psi))[word_vec2, ]   #単語出現率の対数尤度
  LLi0 <- topic_par[w_id2, ] + word_par   #ペアごとの対数尤度
  LLi <- exp(LLi0 - rowMaxs(LLi0))   #尤度に変換
  
  #多項分布より潜在変数zをサンプリング
  z2_rate <- LLi / rowSums(LLi)   #潜在変数zの割当確率
  Zi2 <- rmnom(f2, 1, z2_rate)   #多項分布より潜在変数zをサンプリング
  z2_vec <- as.numeric(Zi2 %*% 1:k2)
  Zi2_T <- t(Zi2)
  
  
  ##パラメータを更新
  #トピック分布のパラメータを更新
  pair_sums0 <- Zi2_T[, pair_id2] %*% Zi1[pair_id1, ]
  pair_sums <- pair_sums0 + alpha02   #ディリクレ分布のパラメータ
  theta2 <- extraDistr::rdirichlet(k2, pair_sums)   #ディリクレ分布からパラメータをサンプリング
  
  
  #単語分布のパラメータを更新
  tsum0 <- matrix(0, nrow=k2, ncol=v2)
  for(j in 1:v2){
    tsum0[, j] <- Zi2_T[, w_list2[[j]], drop=FALSE] %*% w_vec2[[j]]
  }
  tsum <- tsum0 + beta02   #ディリクレ分布のパラメータ
  psi <- extraDistr::rdirichlet(k2, tsum)   #ディリクレ分布からパラメータをサンプリング

  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    PHI[, , mkeep] <- phi
    PSI[, , mkeep] <- psi
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      SEG1 <- SEG1 + Zi1
      SEG2 <- SEG2 + Zi2
    }
    
    #サンプリング結果を確認
    if(rp%%disp==0){
      LL1 <- sum(log(rowSums(par1$Bur)))
      LL2 <- sum(log(rowSums(exp(word_par) * Zi2)))
      LL <- LL1 + LL2   #対数尤度
      print(rp)
      print(c(LL, LLst, LL1, LLst1, LL2, LLst2))
      print(round(rbind(theta1[1:5, ], thetat1[1:5, ]), 3))
      print(round(cbind(psi[, 1:10], psit[, 1:10]), 3))
    }
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep   #バーンイン期間
RS <- R/keep

##サンプリング結果の可視化
#名詞のトピック分布の可視化
matplot(t(THETA1[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[2, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[4, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA1[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#名詞の単語分布の可視化
matplot(t(PHI[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI[2, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI[4, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PHI[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

#下位階層のトピック分布の可視化
matplot(t(THETA2[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[10, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(THETA2[15, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
        
#下位階層の単語分布の可視化
matplot(t(PSI[1, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PSI[2, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PSI[3, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PSI[4, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PSI[5, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")
matplot(t(PSI[6, , ]), type="l", xlab="サンプリング数", ylab="パラメータ")

##サンプリング結果の要約推定量
#トピック割当の事後分布の要約
seg_rate1 <- SEG1 / rowSums(SEG1)
seg_rate2 <- SEG2 / rowSums(SEG2)
cbind(Z1 %*% 1:k1, round(SEG1 / rowSums(SEG1), 3))   #名詞トピックの割当
cbind(Z2 %*% 1:k2, round(SEG2 / rowSums(SEG2), 3))   #動詞トピックの割当

#動詞と名詞トピックの関連
word_theta0 <- matrix(0, nrow=v2, ncol=k1)
for(j in 1:k1){
  word_theta0[, j] <- tapply(seg_rate1[pair_id1, j], word_vec2[pair_id2], sum)
}
round(word_theta <- word_theta0 / rowSums(word_theta0), 3)


#名詞トピック分布の事後推定量
topic_mu1 <- apply(THETA1[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(cbind(topic_mu1, thetat1), 3)
round(topic_sd1 <- apply(THETA1[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#単語割当の事後推定量
word_mu1 <- apply(PHI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(t(rbind(word_mu1, phit)), 3)

#動詞トピック分布の事後推定量
topic_mu2 <- apply(THETA2[, , burnin:(R/keep)], c(1, 2), mean)   #トピック分布の事後平均
round(rbind(topic_mu2, thetat2), 3)
round(topic_sd2 <- apply(THETA2[, , burnin:(R/keep)], c(1, 2), sd), 3)   #トピック分布の事後標準偏差

#下位階層の単語出現確率の事後推定量
word_mu2 <- apply(PSI[, , burnin:(R/keep)], c(1, 2), mean)   #単語の出現率の事後平均
round(cbind(t(word_mu2), t(psit)), 3)




