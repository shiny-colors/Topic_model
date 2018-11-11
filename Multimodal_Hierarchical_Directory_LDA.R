#####マルチモーダル階層ディレクトリLDAモデル#####
options(warn=0)
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(data.table)
library(bayesm)
library(HMM)
library(stringr)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)
#set.seed(2506787)

####データの発生####
##データの設定
s <- 4   #マルチモーダル数
k1 <- 10   #一般語のトピック数
k2 <- 15   #ディレクトリのトピック数
dir <- 50   #ディレクトリ数
d <- 7500   #文書数
v1 <- 500   #ディレクトリ構造に関係のない語彙数
v2 <- 500    #ディレクトリ構造に関係のある語彙数
v <- v1 + v2   #総語彙数
w <- rpois(d, rgamma(d, 75, 0.5))   #文書あたりの単語数
f <- sum(w)   #総単語数

##IDの設定
content_allocation1 <- matrix(1:(s*k1), nrow=s, ncol=k1, byrow=T)
content_allocation2 <- matrix(1:(s*k2), nrow=s, ncol=k2, byrow=T)
d_id <- rep(1:d, w)
t_id <- c()
for(i in 1:d){
  t_id <- c(t_id, 1:w[i])
}

##ディレクトリの割当を設定
dir_freq <- rtpois(d, 1.0, 0, 5)   #文書あたりのディレクトリ数
max_freq <- max(dir_freq)
dir_id <- rep(1:d, dir_freq)   #ディレクトリのid
dir_n <- length(dir_id)
dir_index <- list()
for(i in 1:d){
  dir_index[[i]] <- which(dir_id==i)
}

#ディレクトリの生成
dir_prob <- as.numeric(extraDistr::rdirichlet(1, rep(2.5, dir)))
dir_data <- matrix(0, nrow=dir_n, ncol=dir)

for(i in 1:d){
  repeat{
    x <- rmnom(dir_freq[i], 1, dir_prob)
    if(max(colSums(x))==1){
      index <- dir_index[[i]]
      x <- x[order(as.numeric(x %*% 1:dir)), , drop=FALSE]
      dir_data[index, ] <- x
      break
    }
  }
}
#ディレクトリをベクトルに変換
dir_vec <- as.numeric(dir_data %*% 1:dir)

##コンテンツの生成
#パラメータの設定
alpha <- c(3.0, 2.0, 10.0, 5.0)
prob <- extraDistr::rdirichlet(d, alpha)   #コンテンツの割当確率

#文書ごとにコンテンツを生成
content_list <- list()
for(i in 1:d){
  x <- rmnom(w[i], 1, prob[i, ])
  content_list[[i]] <- sort(x %*% 1:s)
}
content_vec <- unlist(content_list)
content_dir <- content_vec[rep(1:f, rep(dir_freq, w))]
content_id <- rep(1:d, dir_freq*w)

##パラメータの設定
#ディリクレ分布の事前分布
alpha11 <- rep(0.15, k1)
alpha12 <- rep(0.10, k2)
alpha21 <- c(rep(0.075, length(1:v1)), rep(0.000001, length(1:v2)))
alpha22 <- c(rep(0.000001, length(1:v1)), rep(0.05, length(1:v2)))
beta1 <- c(10.0, 8.0)

##すべての単語が出現するまでデータの生成を続ける
for(rp in 1:1000){
  print(rp)
  
  #ディリクレ分布からパラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(d, alpha11)
  theta2 <- thetat2 <- extraDistr::rdirichlet(dir, alpha12)
  phi1 <- phi2 <- list()
  for(j in 1:s){
    phi1[[j]] <- extraDistr::rdirichlet(k1, alpha21)
    phi2[[j]] <- extraDistr::rdirichlet(k2, alpha22)
  }
  phit1 <- phi1; phit2 <- phi2
  phi1_data <- do.call(rbind, phi1); phi2_data <- do.call(rbind, phi2)
  lambda1 <- lambdat1 <- rbeta(d, beta1[1], beta1[2])
  lambda2 <- lambdat2 <-  c(0.6, 0.7, 0.5, 0.6)
  
  #スイッチング変数を生成
  gamma_list <- list()
  for(i in 1:d){
    if(dir_freq[i]==1){
      gamma_list[[i]] <- 1
    } else {
      par <- runif(dir_freq[i], 1.0, 4.5)
      gamma_list[[i]] <- as.numeric(extraDistr::rdirichlet(1, par))
    }
  }
  
  ##モデルに基づきデータを生成
  word_list <- wd_list <- Z11_list <- Z12_list <- Z21_list <- Z22_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    #多項分布から文書のスイッチング変数を生成
    r1 <- (lambda1[i]+lambda2[content_list[[i]]]) / 2; r0 <- ((1-lambda1[i])+(1-lambda2[content_list[[i]]])) / 2
    prob <- r1 / (r1 + r0)
    z11_vec <- rbinom(w[i], 1, prob)
    index_z11 <- which(z11_vec==1)
    
    #多項分布からディレクトリのスイッチング変数を生成
    n <- dir_freq[i]
    if(dir_freq[i]==1){
      z12 <- rep(1, w[i])
      Z12_list[[i]] <- z12
      z12_vec <- as.numeric((Z12_list[[i]] * matrix(dir_vec[dir_index[[i]]], nrow=w[i], ncol=n, byrow=T)) %*% rep(1, n))
    } else {
      Z12_list[[i]] <- rmnom(w[i], 1, gamma_list[[i]])
      z12_vec <- as.numeric((Z12_list[[i]] * matrix(dir_vec[dir_index[[i]]], nrow=w[i], ncol=n, byrow=T)) %*% rep(1, n))
    }
    
    #多項分布より一般語のトピックを生成
    z21 <- matrix(0, nrow=w[i], ncol=k1)
    z21[-index_z11, ] <- rmnom(w[i]-length(index_z11), 1, theta1[i, ])
    z21_vec <- as.numeric(z21 %*% 1:k1)
    
    #多項分布よりディレクトリのトピックを生成
    z22 <- matrix(0, nrow=w[i], ncol=k2)
    z22[index_z11, ] <- rmnom(length(index_z11), 1, theta2[z12_vec[index_z11], ])
    z22_vec <- as.numeric(z22 %*% 1:k2)
    
    #トピックおよびディレクトリから単語を生成
    word <- matrix(0, nrow=w[i], ncol=v)
    index_row1 <- rowSums(content_allocation1[content_list[[i]][-index_z11], ] * z21[-index_z11, ])
    index_row2 <- rowSums(content_allocation2[content_list[[i]][index_z11], ] * z22[index_z11, ])
    word[-index_z11, ] <- rmnom(w[i]-length(index_z11), 1, phi1_data[index_row1, ])   #トピックから単語を生成
    word[index_z11, ] <- rmnom(length(index_z11), 1, phi2_data[index_row2, ])   #ディレクトリから単語を生成
    wd <- as.numeric(word %*% 1:v)
    storage.mode(word) <- "integer"
    
    #データを格納
    Z11_list[[i]] <- z11_vec
    Z21_list[[i]] <- z21
    Z22_list[[i]] <- z22
    wd_list[[i]] <- wd
    word_list[[i]] <- word
    WX[i, ] <- colSums(word)
  }
  #全単語が出現していたらbreak
  if(min(colSums(WX) > 0)) break
}

##リストを変換
wd <- unlist(wd_list)
Z11 <- unlist(Z11_list)
z12_list <- list()
for(i in 1:d){
  z <- matrix(0, nrow=w[i], ncol=max_freq)
  z[, 1:dir_freq[i]] <- Z12_list[[i]] 
  z12_list[[i]] <- z
}
Z12 <- do.call(rbind, z12_list)
Z21 <- do.call(rbind, Z21_list)
Z22 <- do.call(rbind, Z22_list)
z11_vec <- Z11
z21_vec <- as.numeric(Z21 %*% 1:k1)
z22_vec <- as.numeric(Z22 %*% 1:k2)
sparse_data <- sparseMatrix(i=1:f, j=wd, x=rep(1, f), dims=c(f, v))
sparse_data_T <- t(sparse_data)
rm(word_list); rm(wd_list); rm(Z21_list); rm(Z22_list)
gc(); gc()


##データの設定
#ディレクトリの割当を設定
dir_z <- matrix(0, nrow=d, ncol=dir)
dir_list1 <- dir_list2 <- list()
directory_id_list <- list()
for(i in 1:d){
  dir_z[i, ] <- colSums(dir_data[dir_index[[i]], , drop=FALSE])
  dir_list1[[i]] <- (dir_z[i, ] * 1:dir)[dir_z[i, ] > 0]
  dir_list2[[i]] <- cbind(matrix(dir_list1[[i]], nrow=w[i], ncol=dir_freq[i], byrow=T), 
                          matrix(0, nrow=w[i], ncol=max_freq-dir_freq[i]))
  directory_id_list[[i]] <- rep(paste(dir_list1[[i]], collapse = ",", sep=""), w[i])
}

dir_Z <- dir_z[d_id, ]
dir_matrix <- do.call(rbind, dir_list2)
directory_id <- unlist(directory_id_list)
storage.mode(dir_Z) <- "integer"

#ディレクトリ数ごとにディレクトリを作成
max_freq <- max(dir_freq)
dir_no <- dir_Z * matrix(1:dir, nrow=f, ncol=dir, byrow=T)
freq_index1 <- freq_index2 <- list()
freq_word <- rep(0, max_freq)

for(j in 1:max_freq){
  x <- as.numeric(t(dir_Z * matrix(dir_freq[d_id], nrow=f, ncol=dir)))
  freq_index1[[j]] <- which(dir_freq[d_id]==j)
  freq_index2[[j]] <- which(x[x!=0]==j)
  freq_word[j] <- length(freq_index2[[j]])/j
}
x <- as.numeric(t(dir_no)); dir_v <- x[x!=0]   #ディレクトリ数に合わせたディレクトリベクトル
x <- as.numeric(t(dir_Z * matrix(1:f, nrow=f, ncol=dir))); wd_v <- wd[x[x!=0]]   #ディレクトリ数に合わせた単語ベクトル
vec1 <- rep(1, k1); vec2 <- rep(1, k2)
N <- length(wd_v)
rm(x); rm(dir_no); rm(dir_Z)
gc(); gc()


#####マルコフ連鎖モンテカルロ法でDLDAを推定####
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

##事前分布の設定
alpha1 <- 0.1
alpha2 <- 0.01
beta1 <- 1
beta2 <- 1

##真値の設定
theta1 <- thetat1
theta2 <- thetat2
phi1 <- phit1
phi1_data <- do.call(rbind, phit1)
phi2 <- phit2
phi2_data <- do.call(rbind, phit2)
lambda1 <- lambdat1
lambda2 <- lambdat2
gamma <- matrix(0, nrow=d, ncol=max_freq)
for(i in 1:d){
  gamma[i, 1:dir_freq[i]] <- gamma_list[[i]]
}
gammat <- gamma

##初期値の設定
#トピック分布の初期値
theta1 <- extraDistr::rdirichlet(d, rep(10.0, k1))
theta2 <- extraDistr::rdirichlet(dir, rep(10.0, k2))
phi1 <- phi2 <- list()
for(j in 1:s){
  phi1[[j]] <- extraDistr::rdirichlet(k1, rep(10.0, v))
  phi2[[j]] <- extraDistr::rdirichlet(k2, rep(10.0, v))
}
phi1_data <- do.call(rbind, phi1); phi2_data <- do.call(rbind, phi2)


#スイッチング分布の初期値
lambda1 <- rep(0.5, d); lambda2 <- rep(0.5, s)
gamma <- matrix(0, nrow=d, ncol=max_freq)
for(i in 1:d){
  if(dir_freq[i]==1){
    gamma[i, 1] <- 1
  } else {
    gamma[i, 1:dir_freq[i]] <- as.numeric(extraDistr::rdirichlet(1, rep(10.0, dir_freq[i])))
  }
}

##パラメータの保存用配列
THETA1 <- array(0, dim=c(d, k1, R/keep))
THETA2 <- array(0, dim=c(dir, k2, R/keep))
PHI1 <- array(0, dim=c(k1*s, v, R/keep))
PHI2 <- array(0, dim=c(k2*s, v, R/keep))
GAMMA <- array(0, dim=c(d, max_freq, R/keep))
LAMBDA1 <- matrix(0, nrow=R/keep, ncol=d)
LAMBDA2 <- matrix(0, nrow=R/keep, ncol=s)
SEG11 <- rep(0, f)
SEG12 <- matrix(0, nrow=f, ncol=max_freq)
SEG21 <- matrix(0, nrow=f, ncol=k1)
SEG22 <- matrix(0, nrow=f, ncol=k2)
storage.mode(SEG11) <- "integer"
storage.mode(SEG12) <- "integer"
storage.mode(SEG21) <- "integer"
storage.mode(SEG22) <- "integer"


##インデックスを設定
#文書と単語のインデックスを作成
doc_list1 <- doc_list2 <- doc_vec1 <- doc_vec2 <- list()
wd_list1 <- wd_list2 <- wd_vec1 <- wd_vec2 <- list()
dir_list <- dir_vec <- list()
cont_list1 <- cont_list2 <- list()
freq_list <- list()
directory_id0 <- paste(",", directory_id, ",", sep="")

for(i in 1:d){
  doc_list1[[i]] <- which(d_id==i)
  doc_vec1[[i]] <- rep(1, length(doc_list1[[i]]))
}
for(i in 1:dir){
  doc_list2[[i]] <- which(dir_v==i)
  doc_vec2[[i]] <- rep(1, length(doc_list2[[i]]))
  dir_list[[i]] <- which(str_detect(directory_id0, paste(",", as.character(i), ",", sep=""))==TRUE)
  dir_vec[[i]] <- rep(1, length(dir_list[[i]]))
}
for(j in 1:v){
  wd_list1[[j]] <- which(wd==j)
  wd_vec1[[j]] <- rep(1, length(wd_list1[[j]]))
  wd_list2[[j]] <- which(wd_v==j)
  wd_vec2[[j]] <- rep(1, length(wd_list2[[j]]))
}
for(j in 1:s){
  cont_list1[[j]] <- which(content_vec==j)
  cont_list2[[j]] <- which(content_dir==j)
}
for(j in 1:max_freq){
  freq_list[[j]] <- which(dir_freq==j)
}

##対数尤度の基準値
LLst <- sum(sparse_data %*% log(colMeans(sparse_data)))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##単語ごとに文書スイッチング変数を生成
  #トピックとディレクトリの期待尤度
  Lho1 <- matrix(0, nrow=f, ncol=k1); Lho2 <- matrix(0, nrow=N, ncol=k2)
  for(j in 1:s){
    Lho1[cont_list1[[j]], ] <- theta1[d_id[cont_list1[[j]]], ] * t(phi1_data[content_allocation1[j, ], ])[wd[cont_list1[[j]]], ]
    Lho2[cont_list2[[j]], ] <- theta2[dir_v[cont_list2[[j]]], ] * t(phi2_data[content_allocation2[j, ], ])[wd_v[cont_list2[[j]]], ]
  }
  Li1 <- as.numeric(Lho1 %*% vec1)   #トピックの期待尤度
  LLi0 <- matrix(0, nrow=f, ncol=max_freq)   #ディレクトリの期待尤度
  for(j in 1:max_freq){
    LLi0[freq_index1[[j]], 1:j] <- matrix(Lho2[freq_index2[[j]], ] %*% vec2, nrow=freq_word[j], ncol=j, byrow=T)
  }
  LLi2 <- gamma[d_id, ] * LLi0
  Li2 <- as.numeric(LLi2 %*% rep(1, max_freq))
  rm(LLi0)

  #ベルヌーイ分布よりスイッチング変数を生成
  r1 <- (lambda1[d_id]+lambda2[content_vec]) / 2; r0 <- ((1-lambda1[d_id])+(1-lambda2[content_vec])) / 2
  switching_prob <- r1*Li2 / (r1*Li2 + r0*Li1)
  Zi11 <- rbinom(f, 1, switching_prob)   #スイッチング変数をサンプリング
  index_z11 <- which(Zi11==1)
  
  ##単語ごとにディレクトリのスイッチング変数をサンプリング
  switching_prob <- LLi2[index_z11, ] / as.numeric(LLi2[index_z11, ] %*% rep(1, max_freq))   #ディレクトリの割当確率
  Zi12 <- matrix(0, nrow=f, ncol=max_freq)
  Zi12[index_z11, ] <- rmnom(length(index_z11), 1, switching_prob)   #スイッチング変数をサンプリング
  Zi12_T <- t(Zi12)
  
  ##混合率をサンプリング
  #文書のスイッチング変数の混合率をサンプリング
  for(i in 1:d){
    s1 <- sum(Zi11[doc_list1[[i]]])
    v1 <- w[i] - s1 
    lambda1[i] <- rbeta(1, s1 + beta1, v1 + beta2)   #ベータ分布から混合率をサンプリング
  }
  for(j in 1:s){
    s2 <- sum(Zi11[cont_list1[[j]]])
    v2 <- length(cont_list1[[j]]) - s2
    lambda2[j] <- rbeta(1, s2 + beta1, v2 + beta2)   #ベータ分布から混合率をサンプリング
  }
  
  #ディレクトリのスイッチング変数の混合率をサンプリング
  dsum0 <- matrix(0, nrow=d, ncol=max_freq)
  for(i in 1:d){
    if(dir_freq[i]==1) next
    dsum0[i, ] <- Zi12_T[, doc_list1[[i]]] %*% doc_vec1[[i]]
  }
  for(j in 2:max_freq){
    gamma[freq_list[[j]], 1:j] <- extraDistr::rdirichlet(length(freq_list[[j]]), dsum0[freq_list[[j]], 1:j] + alpha1)
  }
  gamma[freq_list[[1]], 1] <- 1 
  
  
  ##一般語トピックをサンプリング
  Zi21 <- matrix(0, nrow=f, ncol=k1)
  z_rate <- Lho1[-index_z11, ] / Li1[-index_z11]   #トピックの割当確率
  Zi21[-index_z11, ] <- rmnom(f-length(index_z11), 1, z_rate)   #トピックをサンプリング
  Zi21_T <- t(Zi21)
  
  ##ディレクトリトピックをサンプリング
  #ディレクトリのトピック尤度を設定
  index <- as.numeric((Zi12 * dir_matrix) %*% rep(1, max_freq))
  Lho2 <- matrix(0, nrow=f, ncol=k2)
  for(j in 1:s){
    cont_z11 <- cont_list1[[j]]*Zi11[cont_list1[[j]]]
    Lho2[cont_z11, ] <- theta2[index[cont_list1[[j]]], ] * t(phi2_data[content_allocation2[j, ], ])[wd[cont_z11], ]
  }

  #トピックの割当確率の設定とトピックのサンプリング
  Zi22 <- matrix(0, nrow=f, ncol=k2)
  Lho2_par <- Lho2[index_z11, ]
  z_rate <- Lho2_par / as.numeric((Lho2_par %*% vec2))   #トピックの割当確率
  Zi22[index_z11, ] <- rmnom(nrow(z_rate), 1, z_rate)   #多項分布からトピックをサンプリング
  Zi22_T <- t(Zi22)
  
  ##トピック分布のパラメータをサンプリング
  #一般語のトピック分布のパラメータをサンプリング
  wsum0 <- matrix(0, nrow=d, ncol=k1)
  for(i in 1:d){
    wsum0[i, ] <- Zi21_T[, doc_list1[[i]], drop=FALSE] %*% doc_vec1[[i]]
  }
  wsum <- wsum0 + alpha1   #ディリクレ分布のパラメータ
  theta1 <- extraDistr::rdirichlet(d, wsum)   #パラメータをサンプリング
  
  #ディレクトリのトピック分布のパラメータをサンプリング
  wsum0 <- matrix(0, nrow=dir, ncol=k2)
  for(i in 1:dir){
    x <- z21_vec[dir_list[[i]]]; x[x!=i] <- 0; x[x==i] <- 1
    wsum0[i, ] <- Zi22_T[, dir_list[[i]], drop=FALSE] %*% dir_vec[[i]]
  }
  wsum <- wsum0 + alpha1   #ディリクレ分布のパラメータ
  theta2 <- extraDistr::rdirichlet(dir, wsum)   #パラメータをサンプリング
  
  ##単語分布のパラメータをサンプリング
  #トピックおよびディレクトリの単語分布をサンプリング
  phi1 <- phi2 <- list()
  for(j in 1:s){
    #ディリクレ分布のパラメータ
    vsum1 <- (Zi21_T[, cont_list1[[j]]] %*% sparse_data[cont_list1[[j]], ]) + alpha2
    vsum2 <- (Zi22_T[, cont_list1[[j]]] %*% sparse_data[cont_list1[[j]], ]) + alpha2
    
    #ディリクレ分布からパラメータをサンプリング
    phi1[[j]] <- extraDistr::rdirichlet(k1, vsum1)
    phi2[[j]] <- extraDistr::rdirichlet(k2, vsum2)
  }
  phi1_data <- do.call(rbind, phi1); phi2_data <- do.call(rbind, phi2)

  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    PHI1[, , mkeep] <- phi1_data
    PHI2[, , mkeep] <- phi2_data
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    GAMMA[, , mkeep] <- gamma
    LAMBDA1[mkeep, ] <- lambda1
    LAMBDA2[mkeep, ] <- lambda2
  }  
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG11 <- SEG11 + Zi11
    SEG12 <- SEG12 + Zi12
    SEG21 <- SEG21 + Zi21
    SEG22 <- SEG22 + Zi22
  }
  if(rp%%disp==0){
    #対数尤度を計算
    index <- as.numeric((Zi12 * dir_matrix) %*% rep(1, max_freq))
    Lho1 <- matrix(0, nrow=f, ncol=k1); Lho2 <- matrix(0, nrow=f, ncol=k2)
    for(j in 1:s){
      cont_z10 <- cont_list1[[j]]*(1-Zi11[cont_list1[[j]]])
      cont_z11 <- cont_list1[[j]]*Zi11[cont_list1[[j]]]
      Lho1[cont_z10, ] <- theta1[d_id[cont_z10], ] * t(phi1_data[content_allocation1[j, ], ])[wd[cont_z10], ]
      Lho2[cont_z11, ] <- theta2[index[cont_list1[[j]]], ] * t(phi2_data[content_allocation2[j, ], ])[wd[cont_z11], ]
    }
    Lho <- sum(log(rowSums(Lho1) + rowSums(Lho2)))
        
    #サンプリング結果を確認
    print(rp)
    print(c(Lho, LLst))
    print(c(mean(Zi11), mean(Z11)))
    print(round(rbind(phi1[[2]][, 491:510], phit1[[2]][, 491:510]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 2000/keep
RS <- R/keep

##サンプリング結果の可視化
#トピック分布の可視化
matplot(t(THETA1[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA1[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[25, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA2[50, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#単語分布の可視化
matplot(t(PHI1[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI1[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI1[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI1[7, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[4, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[6, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI2[8, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

##サンプリング結果の事後分布
#トピック分布の事後平均
round(cbind(apply(THETA1[, , burnin:RS], c(1, 2), mean), thetat1), 3)
round(apply(THETA1[, , burnin:RS], c(1, 2), sd), 3)
round(cbind(apply(THETA2[, , burnin:RS], c(1, 2), mean), thetat2), 3)
round(apply(THETA2[, , burnin:RS], c(1, 2), sd), 3)

#単語分布の事後平均
round(cbind(t(apply(PHI1[, , burnin:RS], c(1, 2), mean)), t(phit1)), 3)
round(t(apply(PHI1[, , burnin:RS], c(1, 2), sd)), 3)
round(cbind(t(apply(PHI2[, , burnin:RS], c(1, 2), mean)), t(phit2)), 3)
round(t(apply(PHI2[, , burnin:RS], c(1, 2), sd)), 3)



##潜在変数のサンプリング結果の事後分布
seg11_rate <- SEG11 / max(SEG11); seg12_rate <- SEG12 / max(SEG11)
seg21_rate <- SEG21 / max(rowSums(SEG21))
seg22_rate <- SEG22 / max(rowSums(SEG22))

seg11_rate[is.nan(seg11_rate)] <- 0; seg12_rate[is.nan(seg12_rate)] <- 0
seg21_rate[is.nan(seg21_rate)] <- 0
seg22_rate[is.nan(seg2_rate)] <- 0

#トピック割当結果を比較
round(cbind(SEG11, seg11_rate), 3)
round(cbind(rowSums(SEG12), seg12_rate), 3)
round(cbind(rowSums(SEG21), seg21_rate), 3)
round(cbind(rowSums(SEG22), seg22_rate), 3)