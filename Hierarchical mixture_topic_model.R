#####階層混合トピックモデル#####
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

#set.seed(93441)

####データの発生####
##データの設定
k11 <- 15   #上位階層
k21 <- 10   #主語のトピック数
k22 <- 10   #述語のトピック数
k23 <- 15   #目的語のトピック数
d <- 2000   #文書数
v1 <- 200   #語彙数
v2 <- 400
v3 <- 600
v <- v1 + v2 + v3   #総語彙数
a <- rpois(d, rgamma(d, 25, 0.7))   #区切り文章数
a[a < 5] <- ceiling(runif(sum(a <= 5), 5, 10))
f11 <- sum(a)   #区切り文章総数
w11 <- rtpois(f11, 0.7, 1, Inf)   #区切り文章ごとの単語数
w12 <- rtpois(f11, 1.25, 1, Inf)
w13 <- rtpois(f11, 4, 1, Inf)
f21 <- sum(w11)   #文要素ごとの単語数
f22 <- sum(w12)
f23 <- sum(w13)
f <- f21 + f22 + f23   #総単語数


##文書IDを設定
a_id <- rep(1:d, a)
w_id11 <- rep(1:f11, w11)
w_id12 <- rep(1:f11, w12)
w_id13 <- rep(1:f11, w13)
w21 <- as.numeric(tapply(w11, a_id, sum))
w22 <- as.numeric(tapply(w12, a_id, sum))
w23 <- as.numeric(tapply(w13, a_id, sum))
w_id21 <- rep(1:d, w21)
w_id22 <- rep(1:d, w22)
w_id23 <- rep(1:d, w23)


##モデルに基づきデータを生成
for(rp in 1:1000){
  print(rp)
  
  #ディレクリ分布のパラメータの設定
  alpha11 <- rep(0.05, k11) 
  alpha21 <- rep(0.15, k21)
  alpha22 <- rep(0.1, k22)
  alpha23 <- rep(0.1, k23)
  beta21 <- rep(0.125, v1)
  beta22 <- rep(0.1, v2)
  beta23 <- rep(0.1, v3)
  
  #ディレクリ分布からパラメータを生成
  theta11 <- thetat11 <- extraDistr::rdirichlet(d, alpha11)
  theta21 <- thetat21 <- extraDistr::rdirichlet(k11, alpha21)
  theta22 <- thetat22 <- extraDistr::rdirichlet(k11, alpha22)
  theta23 <- thetat23 <- extraDistr::rdirichlet(k11, alpha23)
  phi1 <- phit1 <- extraDistr::rdirichlet(k21, beta21)
  phi2 <- phit2 <- extraDistr::rdirichlet(k22, beta22)
  phi3 <- phit3 <- extraDistr::rdirichlet(k23, beta23)

  #データを生成
  ZX21 <- matrix(0, nrow=f11, k21)
  ZX22 <- matrix(0, nrow=f11, k22)
  ZX23 <- matrix(0, nrow=f11, k23)
  Z11_list <- list()
  Z21_list <- list()
  Z22_list <- list()
  Z23_list <- list()
  WX1 <- matrix(0, nrow=f11, v1)
  WX2 <- matrix(0, nrow=f11, v2)
  WX3 <- matrix(0, nrow=f11, v3)
  word1_list <- list()
  word2_list <- list()
  word3_list <- list()
  
  for(i in 1:f11){
    if(i%%1000==0){
      print(i)
    }
    
    #文章のトピックを生成
    z11 <- rmnom(1, 1, theta11[a_id[i], ])
    z11_vec <- as.numeric(z11 %*% 1:k11)
    
    #文要素ごとのトピックを生成
    z21 <- rmnom(w11[i], 1, theta21[z11_vec, ])
    z22 <- rmnom(w12[i], 1, theta22[z11_vec, ])
    z23 <- rmnom(w13[i], 1, theta23[z11_vec, ])
    z21_vec <- as.numeric(z21 %*% 1:k21)
    z22_vec <- as.numeric(z22 %*% 1:k22)
    z23_vec <- as.numeric(z23 %*% 1:k23)
    
    #生成したトピックから単語を生成
    word1 <- rmnom(w11[i], 1, phi1[z21_vec, ])
    word2 <- rmnom(w12[i], 1, phi2[z22_vec, ])
    word3 <- rmnom(w13[i], 1, phi3[z23_vec, ])
    
    #データを格納
    ZX21[i, ] <- colSums(z21)
    ZX22[i, ] <- colSums(z22)
    ZX23[i, ] <- colSums(z23)
    Z11_list[[i]] <- z11
    Z21_list[[i]] <- z21
    Z22_list[[i]] <- z22
    Z23_list[[i]] <- z23
    WX1[i, ] <- colSums(word1)
    WX2[i, ] <- colSums(word2)
    WX3[i, ] <- colSums(word3)
    word1_list[[i]] <- word1
    word2_list[[i]] <- word2
    word3_list[[i]] <- word3
  }
  #すべての語彙が出現すればループ終了
  if(min(c(colSums(WX1), colSums(WX2), colSums(WX3))) > 0){
    break
  }
}

#リストを変換
Z11 <- do.call(rbind, Z11_list)
Z21 <- do.call(rbind, Z21_list)
Z22 <- do.call(rbind, Z22_list)
Z23 <- do.call(rbind, Z23_list)
words1 <- do.call(rbind, word1_list)
words2 <- do.call(rbind, word2_list)
words3 <- do.call(rbind, word3_list)

#スパース行列に変換
sparse_wx1 <- as(WX1, "CsparseMatrix")
sparse_wx2 <- as(WX2, "CsparseMatrix")
sparse_wx3 <- as(WX3, "CsparseMatrix")
sparse_word1 <- as(words1, "CsparseMatrix")
sparse_word2 <- as(words2, "CsparseMatrix")
sparse_word3 <- as(words3, "CsparseMatrix")
storage.mode(ZX21) <- "integer"
storage.mode(ZX22) <- "integer"
storage.mode(ZX23) <- "integer"
storage.mode(Z11) <- "integer"
storage.mode(Z21) <- "integer"
storage.mode(Z22) <- "integer"
storage.mode(Z23) <- "integer"
storage.mode(WX1) <- "integer"
storage.mode(WX2) <- "integer"
storage.mode(WX3) <- "integer"
rm(words1); rm(words2); rm(words3)
rm(word1_list); rm(word2_list); rm(word3_list)
rm(Z11_list); rm(Z21_list); rm(Z22_list); rm(Z23_list)
gc(); gc()


####単語ごとのトピックモデルでトピックの初期値を与える####
##単語ごとに尤度と負担率を計算する関数
burden_fr <- function(theta, phi, wd, w, k){
  #負担係数を計算
  Bur <- theta[w, ] * t(phi)[wd, ]   #尤度
  Br <- Bur / rowSums(Bur)   #負担率
  r <- colSums(Br) / sum(Br)   #混合率
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}


##インデックスを設定
#トピックモデルのインデックス
wd1 <- as.numeric(sparse_word1 %*% 1:v1)
wd2 <- as.numeric(sparse_word2 %*% 1:v2)
wd3 <- as.numeric(sparse_word3 %*% 1:v3)
doc_list01 <- doc_list02 <- doc_list03 <- list()
doc_vec01 <- doc_vec02 <- doc_vec03 <- list()
wd_list01 <- wd_list02 <- wd_list03 <- list()
wd_vec01 <- wd_vec02 <- wd_vec03 <- list()

for(i in 1:d){
  doc_list01[[i]] <- which(w_id21==i)
  doc_list02[[i]] <- which(w_id22==i)
  doc_list03[[i]] <- which(w_id23==i)
  doc_vec01[[i]] <- rep(1, length(doc_list01[[i]]))
  doc_vec02[[i]] <- rep(1, length(doc_list02[[i]]))
  doc_vec03[[i]] <- rep(1, length(doc_list03[[i]]))
}
for(j in 1:v1){
  wd_list01[[j]] <- which(wd1==j)
  wd_vec01[[j]] <- rep(1, length(wd_list01[[j]]))
}
for(j in 1:v2){
  wd_list02[[j]] <- which(wd2==j)
  wd_vec02[[j]] <- rep(1, length(wd_list02[[j]]))
}
for(j in 1:v3){
  wd_list03[[j]] <- which(wd3==j)
  wd_vec03[[j]] <- rep(1, length(wd_list03[[j]]))
}

##アルゴリズムの設定
R <- 1000
keep <- 2  
iter <- 0
burnin <- 500
R_keep <- R/keep
burnin_keep <- burnin/keep
disp <- 10
v_max <- max(c(v1, v2, v3))
max_which <- which.max(c(v1, v2, v3))

##事前分布の設定
alpha01 <- 1
beta01 <- 0.1

##初期値の設定
theta01 <- extraDistr::rdirichlet(d, rep(0.5, k21))
theta02 <- extraDistr::rdirichlet(d, rep(0.5, k22))
theta03 <- extraDistr::rdirichlet(d, rep(0.5, k23))
phi01 <- extraDistr::rdirichlet(k21, rep(0.5, v1))
phi02 <- extraDistr::rdirichlet(k22, rep(0.5, v2))
phi03 <- extraDistr::rdirichlet(k23, rep(0.5, v3))

##パラメータの格納用
THETA01 <- matrix(0, nrow=d, ncol=k21)
THETA02 <- matrix(0, nrow=d, ncol=k22)
THETA03 <- matrix(0, nrow=d, ncol=k23)
PHI01 <- array(0, dim=c(k21, v1, R/keep))
PHI02 <- array(0, dim=c(k22, v2, R/keep))
PHI03 <- array(0, dim=c(k23, v3, R/keep))
Z_SEG01 <- matrix(0, nrow=f21, ncol=k21)
Z_SEG02 <- matrix(0, nrow=f22, ncol=k22)
Z_SEG03 <- matrix(0, nrow=f23, ncol=k23)


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##潜在トピックをサンプリング
  #単語ごとにトピックの出現確率を計算
  word_par1 <- burden_fr(theta01, phi01, wd1, w_id21, k21)
  word_par2 <- burden_fr(theta02, phi02, wd2, w_id22, k22)
  word_par3 <- burden_fr(theta03, phi03, wd3, w_id23, k23)
  word_rate1 <- word_par1$Br; word_rate2 <- word_par2$Br; word_rate3 <- word_par3$Br
  
  #多項分布から単語トピックをサンプリング
  Zi1 <- rmnom(f21, 1, word_rate1)   
  Zi2 <- rmnom(f22, 1, word_rate2)
  Zi3 <- rmnom(f23, 1, word_rate3)
  Zi1_T <- t(Zi1); Zi2_T <- t(Zi2); Zi3_T <- t(Zi3)
  
  ##単語トピックのパラメータを更新
  ##ディクレリ分布からthetaをサンプリング
  wsum01 <- matrix(0, nrow=d, ncol=k21)
  wsum02 <- matrix(0, nrow=d, ncol=k22)
  wsum03 <- matrix(0, nrow=d, ncol=k23)
  for(i in 1:d){
    wsum01[i, ] <- Zi1_T[, doc_list01[[i]]] %*% doc_vec01[[i]]
    wsum02[i, ] <- Zi2_T[, doc_list02[[i]]] %*% doc_vec02[[i]]
    wsum03[i, ] <- Zi3_T[, doc_list03[[i]]] %*% doc_vec03[[i]]
  }
  
  #ディクレリ分布のパラメータ
  wsum1 <- wsum01 + alpha01  
  wsum2 <- wsum02 + alpha01
  wsum3 <- wsum03 + alpha01
  
  #パラメータをサンプリング
  theta01 <- extraDistr::rdirichlet(d, wsum1)   
  theta02 <- extraDistr::rdirichlet(d, wsum2)  
  theta03 <- extraDistr::rdirichlet(d, wsum3)  
  
  ##ディクレリ分布からphiをサンプリング
  vf01 <- matrix(0, nrow=k21, ncol=v1)
  vf02 <- matrix(0, nrow=k22, ncol=v2)
  vf03 <- matrix(0, nrow=k23, ncol=v3)
  for(j in 1:v_max){
    if(j <= v1){ 
      vf01[, j] <- Zi1_T[, wd_list01[[j]], drop=FALSE] %*% wd_vec01[[j]]
    }
    if(j <= v2){
      vf02[, j] <- Zi2_T[, wd_list02[[j]], drop=FALSE] %*% wd_vec02[[j]]
    }
    if(j <= v3){
      vf03[, j] <- Zi3_T[, wd_list03[[j]], drop=FALSE] %*% wd_vec03[[j]]
    }
  }
  
  #ディクレリ分布のパラメータ
  vf1 <- vf01 + beta01; vf2 <- vf02 + beta01; vf3 <- vf03 + beta01
  
  #パラメータをサンプリング
  phi01 <- extraDistr::rdirichlet(k21, vf1)   
  phi02 <- extraDistr::rdirichlet(k22, vf2)  
  phi03 <- extraDistr::rdirichlet(k23, vf3)  
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    PHI01[, , mkeep] <- phi01
    PHI02[, , mkeep] <- phi02
    PHI03[, , mkeep] <- phi03
  }
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    THETA01 <- THETA01 + theta01
    THETA02 <- THETA02 + theta02
    THETA03 <- THETA03 + theta03
    Z_SEG01 <- Z_SEG01 + Zi1
    Z_SEG02 <- Z_SEG02 + Zi2
    Z_SEG03 <- Z_SEG03 + Zi3
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL1 <- sum(log(rowSums(word_par1$Bur)))
    LL2 <- sum(log(rowSums(word_par2$Bur)))
    LL3 <- sum(log(rowSums(word_par3$Bur)))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL1+LL2+LL3, LL1, LL2, LL3))
  }
}


####マルコフ連鎖モンテカルロ法で階層混合トピックモデルを推定######
##アルゴリズムの設定
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##インデックスの設定
a_list <- list()
a_vec <- list()
for(i in 1:d){
  a_list[[i]] <- which(a_id==i)
  a_vec[[i]] <- rep(1, length(a_list[[i]]))
}

s_list1 <- s_list2 <- s_list3 <- list()
s_vec1 <- s_vec2 <- s_vec3 <- list()
for(i in 1:f11){
  if(i%%1000==0){
    print(i)
  }
  s_list1[[i]] <- which(w_id11==i)
  s_list2[[i]] <- which(w_id12==i)
  s_list3[[i]] <- which(w_id13==i)
  s_vec1[[i]] <- rep(1, length(s_list1[[i]]))
  s_vec2[[i]] <- rep(1, length(s_list2[[i]]))
  s_vec3[[i]] <- rep(1, length(s_list3[[i]]))
}


##パラメータの事前分布
alpha11 <- 0.25
alpha21 <- 0.5
alpha22 <- 0.5
alpha23 <- 0.5
beta3 <- beta2 <- beta1 <- 0.1

##パラメータの真値
Zi21 <- Z21; Zi22 <- Z22; Zi23 <- Z23
theta11 <- thetat11; theta21 <- thetat21; theta22 <- thetat22; theta23 <- thetat23
phi1 <- phit1; phi2 <- phit2; phi3 <- phit3

##初期値を設定
Zi21 <- rmnom(f21, 1, rep(1.0, k21))
Zi22 <- rmnom(f22, 1, rep(1.0, k22))
Zi23 <- rmnom(f23, 1, rep(1.0, k23))
theta11 <- extraDistr::rdirichlet(d, rep(1.0, k11))
theta21 <- extraDistr::rdirichlet(k11, rep(1.0, k21))
theta22 <- extraDistr::rdirichlet(k11, rep(1.0, k22))
theta23 <- extraDistr::rdirichlet(k11, rep(1.0, k23))
phi1 <- apply(PHI01[, , burnin_keep:R_keep], c(1, 2), mean)
phi2 <- apply(PHI02[, , burnin_keep:R_keep], c(1, 2), mean)
phi3 <- apply(PHI03[, , burnin_keep:R_keep], c(1, 2), mean)


##パラメータの格納用配列(単語分布phiを定数として固定)
THETA11 <- array(0, dim=c(d, k11, R/keep))
THETA21 <- array(0, dim=c(k11, k21, R/keep))
THETA22 <- array(0, dim=c(k11, k22, R/keep))
THETA23 <- array(0, dim=c(k11, k23, R/keep))
PHI1 <- array(0, dim=c(k21, v1, R/keep))
PHI2 <- array(0, dim=c(k22, v2, R/keep))
PHI3 <- array(0, dim=c(k23, v3, R/keep))
Z_SEG11 <- matrix(0, nrow=f11, ncol=k11)
Z_SEG21 <- matrix(0, nrow=f21, ncol=k21)
Z_SEG22 <- matrix(0, nrow=f22, ncol=k22)
Z_SEG23 <- matrix(0, nrow=f23, ncol=k23)


##対数尤度の基準値
LLst_all <- sum(cbind(WX1, WX2, WX3) %*% log(colSums(cbind(WX1, WX2, WX3)) / sum(cbind(WX1, WX2, WX3))))
LLst1 <- sum(WX1 %*% log(colSums(WX1) / sum(WX1)))
LLst2 <- sum(WX2 %*% log(colSums(WX2) / sum(WX2)))
LLst3 <- sum(WX3 %*% log(colSums(WX3) / sum(WX3)))
LLst <- LLst1 + LLst2 + LLst3


####マルコフ連鎖モンテカルロ法でパラメータをサンプリング
for(rp in 1:R){
  
  ##文章単位のトピックをサンプリング
  #単語トピックの対数尤度を計算
  LLi21_T <- t(Zi21 %*% t(log(theta21)))
  LLi22_T <- t(Zi22 %*% t(log(theta22)))
  LLi23_T <- t(Zi23 %*% t(log(theta23)))
  
  #文章ごとに対数尤度の和を取る
  LLi <- matrix(0, nrow=f11, ncol=k11)
  for(i in 1:f11){
    LLi1 <- LLi21_T[, s_list1[[i]]] %*% s_vec1[[i]]
    LLi2 <- LLi22_T[, s_list2[[i]]] %*% s_vec2[[i]]
    LLi3 <- LLi23_T[, s_list3[[i]]] %*% s_vec3[[i]]
    LLi[i, ] <- LLi1 + LLi2 + LLi3 
  }
   
  #潜在変数zの割当確率
  LLi_Z <- log(theta11)[a_id, ] + LLi   #潜在変数の対数尤度
  Li_Z <- exp(LLi_Z - rowMaxs(LLi_Z))   #尤度に変換
  z_rate11 <- Li_Z / rowSums(Li_Z)   #潜在変数の割当確率z
  
  #多項分布よりトピックzをサンプリング
  Zi11 <- rmnom(f11, 1, z_rate11)   #潜在変数をサンプリング
  Zi11_T <- t(Zi11)
  z11_vec <- as.numeric(Zi11 %*% 1:k11)
  
  
  ##パラメータを更新
  ##ディクレリ分布からthetaをサンプリング
  ssum0 <- matrix(0, nrow=d, ncol=k11)
  for(i in 1:d){
    ssum0[i, ] <- Zi11_T[, a_list[[i]]] %*% a_vec[[i]]
  }
  ssum <- ssum0 + alpha11   #ディクレリ分布のパラメータ
  theta11 <- extraDistr::rdirichlet(d, ssum)   #パラメータを生成

  
  ##単語ごとのトピックをサンプリング
  #単語ごとの文章トピックの割当idを設定
  z_id11 <- rep(z11_vec, w11)
  z_id12 <- rep(z11_vec, w12)
  z_id13 <- rep(z11_vec, w13)
  
  #単語ごとにトピックの出現確率を計算
  word_par21 <- burden_fr(theta21, phi1, wd1, z_id11, k21)
  word_par22 <- burden_fr(theta22, phi2, wd2, z_id12, k22)
  word_par23 <- burden_fr(theta23, phi3, wd3, z_id13, k23)
  word_rate21 <- word_par21$Br; word_rate22 <- word_par22$Br; word_rate23 <- word_par23$Br
  
  #多項分布から単語トピックをサンプリング
  Zi21 <- rmnom(f21, 1, word_rate21)   
  Zi22 <- rmnom(f22, 1, word_rate22)
  Zi23 <- rmnom(f23, 1, word_rate23)
  Zi21_T <- t(Zi21); Zi22_T <- t(Zi22); Zi23_T <- t(Zi23)
  
  
  ##単語トピックのパラメータを更新
  ##ディクレリ分布からthetaをサンプリング
  wsum01 <- t(Zi11[rep(1:f11, w11), ]) %*% Zi21
  wsum02 <- t(Zi11[rep(1:f11, w12), ]) %*% Zi22
  wsum03 <- t(Zi11[rep(1:f11, w13), ]) %*% Zi23
  wsum1 <- wsum01 + alpha21; wsum2 <- wsum02 + alpha22; wsum3 <- wsum03 + alpha23   #ディクレリ分布のパラメータ
  
  #パラメータをサンプリング
  theta21 <- extraDistr::rdirichlet(k11, wsum1)   
  theta22 <- extraDistr::rdirichlet(k11, wsum2)  
  theta23 <- extraDistr::rdirichlet(k11, wsum3) 
  
  
  ##ディクレリ分布からphiをサンプリング
  vf01 <- matrix(0, nrow=k21, ncol=v1)
  vf02 <- matrix(0, nrow=k22, ncol=v2)
  vf03 <- matrix(0, nrow=k23, ncol=v3)
  for(j in 1:v_max){
    if(j <= v1){ 
      vf01[, j] <- Zi21_T[, wd_list01[[j]], drop=FALSE] %*% wd_vec01[[j]]
    }
    if(j <= v2){
      vf02[, j] <- Zi22_T[, wd_list02[[j]], drop=FALSE] %*% wd_vec02[[j]]
    }
    if(j <= v3){
      vf03[, j] <- Zi23_T[, wd_list03[[j]], drop=FALSE] %*% wd_vec03[[j]]
    }
  }  
  #ディクレリ分布のパラメータ
  vf1 <- vf01 + beta01; vf2 <- vf02 + beta01; vf3 <- vf03 + beta01
  
  #パラメータをサンプリング
  phi1 <- extraDistr::rdirichlet(k21, vf1)   
  phi2 <- extraDistr::rdirichlet(k22, vf2)  
  phi3 <- extraDistr::rdirichlet(k23, vf3)  
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA11[, , mkeep] <- theta11
    THETA21[, , mkeep] <- theta21
    THETA22[, , mkeep] <- theta22
    THETA23[, , mkeep] <- theta23
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi2
    PHI3[, , mkeep] <- phi3
    
    #トピック割当はバーンイン期間を超えたら格納する
    if(rp%%keep==0 & rp >= burnin){
      Z_SEG11 <- Z_SEG11 + Zi11
      Z_SEG21 <- Z_SEG21 + Zi21
      Z_SEG22 <- Z_SEG22 + Zi22
      Z_SEG23 <- Z_SEG23 + Zi23
    }
    
    if(rp%%disp==0){
      #対数尤度の計算
      LL1 <- sum(log(rowSums(word_par21$Bur)))
      LL2 <- sum(log(rowSums(word_par22$Bur)))
      LL3 <- sum(log(rowSums(word_par23$Bur)))
      LL <- LL1 + LL2 + LL3
      
      #サンプリング結果を確認
      print(rp)
      print(c(LL1+LL2+LL3, LLst, LLst_all, LL1, LLst1, LL2, LLst2, LL3, LLst3))
      print(round(rbind(theta11[1:5, ], thetat11[1:5, ]), 3))
      print(round(cbind(phi1[, 1:10], phit1[, 1:10]), 3))
    }
  }
}

####サンプリング結果の可視化と要約####
##サンプリング結果の可視化
#上位トピック分布の可視化
matplot(t(THETA11[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="文章のトピック分布のサンプリング結果")
matplot(t(THETA11[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="文章のトピック分布のサンプリング結果")
matplot(t(THETA11[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="文章のトピック分布のサンプリング結果")

#下位トピック分布の可視化
matplot(t(THETA21[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA21[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA21[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA22[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA22[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA22[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA23[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA23[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")
matplot(t(THETA23[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="下位のトピック分布のサンプリング結果")

#単語分布の可視化
matplot(t(PHI1[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="主語の単語分布のサンプリング結果")
matplot(t(PHI1[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="主語の単語分布のサンプリング結果")
matplot(t(PHI1[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="主語の単語分布のサンプリング結果")
matplot(t(PHI2[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="述語の単語分布のサンプリング結果")
matplot(t(PHI2[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="述語の単語分布のサンプリング結果")
matplot(t(PHI2[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="述語の単語分布のサンプリング結果")
matplot(t(PHI3[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="目的語の単語分布のサンプリング結果")
matplot(t(PHI3[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="目的語の単語分布のサンプリング結果")
matplot(t(PHI3[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ", main="目的語の単語分布のサンプリング結果")


##サンプリング結果の事後分布の要約

