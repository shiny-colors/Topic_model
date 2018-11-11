#####Switching Directory LDA model#####
options(warn=0)
library(stringr)
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
#set.seed(2506787)


####データの発生####
##データの設定
k1 <- 15   #ディレクトリのトピック数
k2 <- 15   #アイテムのトピック数
k3 <- 7   #一般語のトピック数
a <- 3   #文書の種類数
dir <- 50   #ディレクトリ数
item <- 1000   #アイテム数
v1 <- 350   #ディレクトリの語彙数
v2 <- 350   #アイテムの語彙数
v3 <- 300   #一般語の語彙数
v <- v1 + v2 + v3   #総語彙数
v1_index <- 1:v1
v2_index <- (v1+1):v2
v3_index <- (v2+1):v3


##文書を設定
#文書数とIDの設定
item_freq <- rtpois(item, a=0, b=Inf, rgamma(item, 3.2, 0.5))   #アイテムごとの文書数
dir_freq <- rtpois(sum(item_freq), a=0, b=3, 1.5)   #文書ごとのディレクトリ数
d <- sum(item_freq)   #総文書数
item_id <- rep(1:item, item_freq)   #アイテムID
dir_id <- rep(1:d, dir_freq)   #ディレクトリID

#単語数とIDを設定
w <- rpois(d, rgamma(d, 75, 0.5))   #文書ごとの語彙数
f <- sum(w)   #総単語数
d_id <- rep(1:d, w)   #単語ごとの文書id
t_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))   #単語ごとの文書no
v_id <- rep(item_id, w)   #単語ごとのアイテムid

##ディレクトリの割当を設定
dir_n <- length(dir_id)
dir_index <- list()
for(i in 1:d){
  dir_index[[i]] <- which(dir_id==i)
}

#ディレクトリの生成
dir_prob <- as.numeric(extraDistr::rdirichlet(1, rep(2.5, dir)))
dir_data <- matrix(0, nrow=dir_n, ncol=dir)
dir_d <- matrix(0, nrow=d, ncol=dir)

for(i in 1:d){
  repeat{
    x <- rmnom(dir_freq[i], 1, dir_prob)
    if(max(colSums(x))==1){
      index <- dir_index[[i]]
      x <- x[order(as.numeric(x %*% 1:dir)), , drop=FALSE]
      dir_data[index, ] <- x
      dir_d[i, ] <- colSums(x)
      break
    }
  }
}
#ディレクトリをベクトルに変換
dir_vec <- as.numeric(dir_data %*% 1:dir)

#ディレクトリの単語単位のidを設定
directory_id <- rep("", f)
dir_allocation <- dir_d * matrix(1:dir, nrow=d, ncol=dir, byrow=T)
for(i in 1:d){
  index <- which(d_id==i)
  x <- dir_allocation[i, ]
  directory_id[index] <- paste(x[x!=0], collapse=",", sep="")
}

##パラメータの設定
#ディリクレ分布の事前分布
alpha11 <- rep(0.075, k1)
alpha12 <- rep(0.1, k2)
alpha13 <- rep(5.0, k3)
alpha21 <- c(rep(0.1, v1), rep(0.001, v2), rep(0.001, v3))
alpha22 <- c(rep(0.001, v1), rep(0.1, v2), rep(0.001, v3))
alpha23 <- c(rep(0.001, v1), rep(0.001, v2), rep(0.15, v3))
beta1 <- c(4.75, 3.25, 1.5)

##すべての単語が出現するまでデータの生成を続ける
for(rp in 1:1000){
  print(rp)
  
  #ディリクレ分布からパラメータを生成
  theta1 <- thetat1 <- extraDistr::rdirichlet(dir, alpha11)
  theta2 <- thetat2 <- extraDistr::rdirichlet(item, alpha12)
  theta3 <- thetat3 <- as.numeric(extraDistr::rdirichlet(1, alpha13))
  phi1 <- phit1 <- extraDistr::rdirichlet(k1, alpha21)
  phi2 <- phit2 <- extraDistr::rdirichlet(k2, alpha22)
  phi3 <- phit3 <- extraDistr::rdirichlet(k3, alpha23)
  lambda <- lambdat <- extraDistr::rdirichlet(d, beta1)
  
  #ディレクトリのスイッチング変数のパラメータ
  gamma_list <- list()
  for(i in 1:d){
    if(dir_freq[i]==1){
      gamma_list[[i]] <- 1
    } else {
      par <- c(runif(dir_freq[i], 1.5, 5.0))
      gamma_list[[i]] <- as.numeric(extraDistr::rdirichlet(1, par))
    }
  }
  
  ##モデルに基づきデータを生成
  #データの格納用配列
  Z11_list <- Z12_list <- list()
  Z21_list <- Z22_list <- Z23_list <- list()
  wd_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    #多項分布から文書のスイッチング変数を生成
    z11 <- rmnom(w[i], 1, lambda[i, ])
    z11_vec <- as.numeric(z11 %*% 1:a)
    
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
    
    #多項分布からディレクトリのトピックを生成
    z21 <- matrix(0, nrow=w[i], ncol=k1)
    index1 <- which(z11_vec==1)
    z21[index1, ] <- rmnom(length(index1), 1, theta1[z12_vec[index1], ])
    z21_vec <- as.numeric(z21 %*% 1:k1)
    
    #多項分布からアイテムのトピックを生成
    z22 <- matrix(0, nrow=w[i], ncol=k2)
    index2 <- which(z11_vec==2)
    z22[index2, ] <- rmnom(length(index2), 1, theta2[item_id[i], ])
    z22_vec <- as.numeric(z22 %*% 1:k2)
    
    #多項分布から一般語のトピックを生成
    z23 <- matrix(0, nrow=w[i], ncol=k3)
    index3 <- which(z11_vec==3)
    z23[index3, ] <- rmnom(length(index3), 1, theta3)
    z23_vec <- as.numeric(z23 %*% 1:k3)
    
    
    #生成したトピックから単語を生成
    word <- matrix(0, nrow=w[i], ncol=v)
    if(length(index1) > 0){
      word[index1, ] <- rmnom(length(index1), 1, phi1[z21_vec[index1], ])
    }
    if(length(index2) > 0){
      word[index2, ] <- rmnom(length(index2), 1, phi2[z22_vec[index2], ])
    }
    if(length(index3) > 0){
      word[index3, ] <- rmnom(length(index3), 1, phi3[z23_vec[index3], ])
    }
    wd <- as.numeric(word %*% 1:v)
    
    #データを格納
    Z11_list[[i]] <- z11
    Z21_list[[i]] <- z21
    Z22_list[[i]] <- z22
    Z23_list[[i]] <- z23
    wd_list[[i]] <- wd
    WX[i, ] <- colSums(word)
  }
  if(min(colSums(WX)) > 0) break
}

#リストを変換
wd <- unlist(wd_list)
Z11 <- do.call(rbind, Z11_list); storage.mode(Z11) <- "integer"
Z21 <- do.call(rbind, Z21_list); storage.mode(Z21) <- "integer"
Z22 <- do.call(rbind, Z22_list); storage.mode(Z22) <- "integer"
Z23 <- do.call(rbind, Z23_list); storage.mode(Z23) <- "integer"
z11_vec <- as.numeric(Z11 %*% 1:a)
z21_vec <- as.numeric(Z21 %*% 1:k1)
z22_vec <- as.numeric(Z22 %*% 1:k2)
z23_vec <- as.numeric(Z23 %*% 1:k3)
gamma <- gammat <- matrix(0, nrow=d, ncol=a)
for(i in 1:d){
  gamma[i, 1:dir_freq[i]] <- gammat[i, 1:dir_freq[i]] <- gamma_list[[i]]
}
sparse_data <- sparseMatrix(i=1:f, wd, x=rep(1, f), dims=c(f, v))
sparse_data_T <- t(sparse_data)
rm(Z11_list); rm(Z21_list); rm(Z22_list); rm(Z23_list)
gc(); gc()

##データの設定
#ディレクトリの割当を設定
dir_z <- matrix(0, nrow=d, ncol=dir)
dir_list1 <- dir_list2 <- list()
for(i in 1:d){
  dir_z[i, ] <- colSums(dir_data[dir_index[[i]], , drop=FALSE])
  dir_list1[[i]] <- (dir_z[i, ] * 1:dir)[dir_z[i, ] > 0]
  dir_list2[[i]] <- cbind(matrix(dir_list1[[i]], nrow=w[i], ncol=dir_freq[i], byrow=T), matrix(0, nrow=w[i], ncol=a-dir_freq[i]))
}
dir_Z <- dir_z[d_id, ]
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
dir_matrix <- do.call(rbind, dir_list2)
vec1 <- rep(1, k1); vec2 <- rep(1, k2)
N <- length(wd_v)
rm(x); rm(dir_no)
gc(); gc()


#####マルコフ連鎖モンテカルロ法でSwitching Directory LDA modelを推定####
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
alpha01 <- 0.1
alpha02 <- 0.01
beta01 <- 0.1
beta02 <- 0.05

##パラメータの真値
lambda <- lambdat
gamma <- gammat
theta1 <- thetat1
theta2 <- thetat2
theta3 <- thetat3
phi1 <- phit1
phi2 <- phit2
phi3 <- phit3


##パラメータの初期値を設定
#スイッチング変数の初期値
lambda <- extraDistr::rdirichlet(d, rep(10.0, a))
gamma <- matrix(0, nrow=d, ncol=a)
for(i in 1:d){
  if(dir_freq[i]==1){
    gamma[i, 1] <- 1
  } else {
    gamma[i, 1:dir_freq[i]] <- as.numeric(extraDistr::rdirichlet(1, rep(10.0, dir_freq[i])))
  }
}
#トピック分布の初期値
theta1 <- extraDistr::rdirichlet(dir, rep(10.0, k1))
theta2 <- extraDistr::rdirichlet(item, rep(10.0, k2))
theta3 <- as.numeric(extraDistr::rdirichlet(1, rep(10.0, k3)))

#単語分布の初期値
phi1 <- extraDistr::rdirichlet(k1, rep(5.0, v))
phi2 <- extraDistr::rdirichlet(k2, rep(5.0, v))
phi3 <- extraDistr::rdirichlet(k3, rep(5.0, v))


##パラメータの格納用配列
LAMBDA <- array(0, dim=c(d, a, R/keep))
THETA1 <- array(0, dim=c(dir, k1, R/keep))
THETA2 <- array(0, dim=c(item, k2, R/keep))
THETA3 <- matrix(0, nrow=R/keep, ncol=k3)
PHI1 <- array(0, dim=c(k1, v, R/keep))
PHI2 <- array(0, dim=c(k2, v, R/keep))
PHI3 <- array(0, dim=c(k3, v, R/keep))
SEG11 <- matrix(0, nrow=f, ncol=a)
SEG12 <- matrix(0, nrow=f, ncol=max_freq)
SEG21 <- matrix(0, nrow=f, ncol=k1)
SEG22 <- matrix(0, nrow=f, ncol=k2)
SEG23 <- matrix(0, nrow=f, ncol=k3)

##インデックスを設定
#文書と単語のインデックスを設定
dir_list <- dir_vec <- list()
freq_list <- list()
directory_id0 <- paste(",", directory_id, ",", sep="")

for(i in 1:dir){
  dir_list[[i]] <- which(str_detect(directory_id0, paste(",", as.character(i), ",", sep=""))==TRUE)
  dir_vec[[i]] <- rep(1, length(dir_list[[i]]))
}
for(j in 1:max_freq){
  freq_list[[j]] <- which(dir_freq==j)
}
d_vec <- sparseMatrix(sort(d_id), 1:f, x=rep(1, f), dims=c(d, f))
v_vec <- sparseMatrix(sort(v_id), 1:f, x=rep(1, f), dims=c(item, f))
vec1 <- rep(1, k1); vec2 <- rep(1, k2); vec3 <- rep(1, k3)

##対数尤度の基準値
LLst <- sum(sparse_data %*% log(colSums(sparse_data) / f))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##文書の尤度と期待尤度を設定
  #ディレクトリ、アイテム、一般語の尤度
  Lho1 <- theta1[dir_v, ] * t(phi1)[wd_v, ]   #ディレクトリの尤度
  Lho2 <- theta2[v_id, ] * t(phi2)[wd, ]   #アイテムの尤度
  Lho3 <- matrix(theta3, nrow=f, ncol=k3, byrow=T) * t(phi3)[wd, ]   #一般語の尤度
  
  #ディレクトリの期待尤度
  LLi0 <- matrix(0, nrow=f, ncol=max_freq)
  for(j in 1:max_freq){
    LLi0[freq_index1[[j]], 1:j] <- matrix(Lho1[freq_index2[[j]], ] %*% vec1, nrow=freq_word[[j]], ncol=j, byrow=T)
  }
  LLi1 <- gamma[d_id, ] * LLi0
  par_dir <- as.numeric(LLi1 %*% rep(1, a))   #ディレクトリの期待尤度
  
  #アイテムと一般語の期待尤度
  par_item <- as.numeric(Lho2 %*% vec2)
  par_general <- as.numeric(Lho3 %*% vec3)
  par <- cbind(par_dir, par_item, par_general)
  
  
  ##文書スイッチング変数をサンプリング
  #多項分布から文書スイッチング変数をサンプリング
  lambda_r <- lambda[d_id, ]   #スイッチング変数の混合率
  par_r <- lambda_r * par
  doc_prob <- par_r / rowSums(par_r)   #文書スイッチング変数の割当確率
  Zi11 <- rmnom(f, 1, doc_prob)   #文書スイッチング変数をサンプリング
  
  #割当ごとのインデックスを作成
  index_dir <- which(Zi11[, 1]==1)
  index_item <- which(Zi11[, 2]==1)
  index_general <- which(Zi11[, 3]==1)
  
  
  ##ディレクトリのスイッチング変数をサンプリング
  #多項分布からディレクトリスイッチング変数をサンプリング
  LLi_dir <- LLi1[index_dir, ]
  dir_prob <- LLi_dir / as.numeric(LLi_dir %*% rep(1, a))   #ディレクトリスイッチング変数の割当確率
  Zi12 <- matrix(0, nrow=f, ncol=a)
  Zi12[index_dir, ] <- rmnom(length(index_dir), 1, dir_prob)   #ディレクトリスイッチング変数をサンプリング
  
  
  ##ディリクレ分布から混合率をサンプリング
  #文書の混合率をサンプリング
  rsum <- d_vec %*% Zi11 + beta01   #ディリクレ分布のパラメータ
  lambda <- extraDistr::rdirichlet(d, rsum)   #文書の混合率をサンプリング
  
  #ディレクトリの混合率をサンプリング
  dsum <- d_vec %*% Zi12 + beta02
  for(j in 2:max_freq){   #ディレクトリの混合率をサンプリング
    gamma[freq_list[[j]], 1:j] <- extraDistr::rdirichlet(length(freq_list[[j]]), dsum[freq_list[[j]], 1:j])
  }

  
  ##ディレクトリのトピックをサンプリング
  #ディレクトリのトピック尤度を設定
  index <- as.numeric((Zi12 * dir_matrix) %*% rep(1, max_freq))
  Lho1 <- theta1[index[index_dir], ] * t(phi1)[wd[index_dir], ]   #ディレクトリの尤度
  
  #多項分布からトピックをサンプリング
  Zi21 <- matrix(0, nrow=f, ncol=k1)
  dir_prob <- Lho1 / as.numeric(Lho1 %*% vec1)   #ディレクトリのトピックの割当確率
  Zi21[index_dir, ] <- rmnom(length(index_dir), 1, dir_prob)   #多項分布からトピックをサンプリング
  z21_vec <- as.numeric((Zi12 * dir_matrix) %*% rep(1, max_freq))   #ディレクトリ割当
  Zi21_T <- t(Zi21)
  
  ##アイテムのトピックをサンプリング
  #多項分布からトピックをサンプリング
  Zi22 <- matrix(0, nrow=f, ncol=k2)
  item_prob <- (Lho2 / as.numeric(Lho2 %*% vec2))[index_item, ]   #アイテムのトピックの割当確率
  Zi22[index_item, ] <- rmnom(length(index_item), 1, item_prob)   #多項分布からトピックをサンプリング
  
  ##一般語のトピックをサンプリング
  #多項分布からトピックをサンプリング
  Zi23 <- matrix(0, nrow=f, ncol=k3)
  general_prob <- (Lho3 / as.numeric(Lho3 %*% vec3))[index_general, ]   #一般語のトピックの割当確率
  Zi23[index_general, ] <- rmnom(length(index_general), 1, general_prob)   #多項分布からトピックをサンプリング
  
  
  ##トピック分布のパラメータをサンプリング
  #ディレクトリのトピック分布をサンプリング
  dsum0 <- matrix(0, nrow=dir, ncol=k1)
  for(i in 1:dir){
    x <- z21_vec[dir_list[[i]]]
    index <- which(x==i); x[-index] <- 0; x[index] <- 1   
    dsum0[i, ] <- Zi21_T[, dir_list[[i]], drop=FALSE] %*% x
  }
  dsum <- dsum0 + alpha01   #ディリクレ分布のパラメータ
  theta1 <- extraDistr::rdirichlet(dir, dsum)   #パラメータをサンプリング
  
  #アイテムのトピック分布をサンプリング
  rsum <- v_vec %*% Zi22 + alpha01   #ディリクレ分布のパラメータ
  theta2 <- extraDistr::rdirichlet(item, rsum)   #パラメータをサンプリング
  
  #一般語のトピック分布をサンプリング
  gsum <- rep(1, length(index_general)) %*% Zi23[index_general, , drop=FALSE] + alpha01
  theta3 <- as.numeric(extraDistr::rdirichlet(1, gsum))   #パラメータをサンプリング
  
  
  ##単語分布のパラメータをサンプリング
  #ディリクレ分布のパラメータを推定
  wsum1 <- t(sparse_data_T %*% Zi21) + alpha02
  wsum2 <- t(sparse_data_T %*% Zi22) + alpha02
  wsum3 <- t(sparse_data_T %*% Zi23) + alpha02
  
  #ディリクレ分布からパラメータをサンプリング
  phi1 <- extraDistr::rdirichlet(k1, wsum1)
  phi2 <- extraDistr::rdirichlet(k2, wsum2)
  phi3 <- extraDistr::rdirichlet(k3, wsum3)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    LAMBDA[, , mkeep] <- lambda
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi2
    PHI3[, , mkeep] <- phi3
    THETA1[, , mkeep] <- theta1
    THETA2[, , mkeep] <- theta2
    THETA3[mkeep, ] <- theta3
  }  

  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG11 <- SEG11 + Zi11
    SEG12 <- SEG12 + Zi12
    SEG21 <- SEG21 + Zi21
    SEG22 <- SEG22 + Zi22
    SEG23 <- SEG23 + Zi23
  }
  
  #対数尤度の計算とサンプリング結果の確認
  if(rp%%disp==0){
    #対数尤度を計算
    index <- which(z21_vec > 0)
    LL1 <- sum(log((theta1[z21_vec, ] * t(phi1)[wd[index], ]) %*% vec1))   #ディレクトリの対数尤度
    LL2 <- sum(log((theta2[v_id, ] * t(phi2)[wd, ])[index_item, ] %*% vec2))   #アイテムの対数尤度
    LL3 <- sum(log((matrix(theta3, nrow=f, ncol=k3, byrow=T) * t(phi3)[wd, ])[index_general, ] %*% vec3))   #一般語の対数尤度
    LL <- sum(LL1 + LL2 + LL3)   #対数尤度の総和
    
    #サンプリング結果の確認
    print(rp)
    print(c(LL, LLst))
    print(round(rbind(theta1[1:7, ], thetat1[1:7, ]), 3))
  }
}

round(phi1[, 1:10], 3)
