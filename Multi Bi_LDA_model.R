#####Switching Multinom LDA model#####
options(warn=0)
library(MASS)
library(lda)
library(RMeCab)
library(matrixStats)
library(Matrix)
library(data.table)
library(bayesm)
library(HMM)
library(extraDistr)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)
#set.seed(2506787)

####データの発生####
##データの設定
r <- 5   #評価スコア数
s <- 3   #極性値数
a <- 3   #分岐数
k11 <- 5   #ユーザーの評価スコアのトピック数
k12 <- 5   #アイテムの評価スコアのトピック数
K1 <- matrix(1:(k11*k12), nrow=k11, ncol=k12, byrow=T)   #トピックの配列
k21 <- 10   #ユーザーのテキストのトピック数
k22 <- 15   #アイテムのテキストのトピック数
hh <- 1000   #レビュアー数
item <- 200   #アイテム数
v1 <- 300   #評価スコアの語彙数
v2 <- 350   #ユーザートピックの語彙数
v3 <- 350   #アイテムトピックの語彙数
v <- v1 + v2 + v3   #総語彙数
spl <- matrix(1:v1, nrow=s, ncol=v1/s, byrow=T)
v1_index <- 1:v1
v2_index <- (v1+1):v2
v3_index <- (v2+1):v

##IDと欠損ベクトルの作成
#IDを仮設定
user_id0 <- rep(1:hh, rep(item, hh))
item_id0 <- rep(1:item, hh)

#欠損ベクトルを作成
for(rp in 1:100){
  m_vec <- rep(0, hh*item)
  for(i in 1:item){
    prob <- runif(1, 0.025, 0.16)
    m_vec[item_id0==i] <- rbinom(hh, 1, prob)
  }
  m_index <- which(m_vec==1)
  
  #完全なIDを設定
  user_id <- user_id0[m_index]
  item_id <- item_id0[m_index]
  d <- length(user_id)   #総レビュー数
  
  #すべてのパターンが生成されればbreak
  if(length(unique(user_id))==hh & length(unique(item_id))==item) break
}

#単語数を設定
w <- rpois(d, rgamma(d, 25, 0.5))   #文書あたりの単語数
f <- sum(w)   #総単語数
n_user <- plyr::count(user_id)$freq
n_item <- plyr::count(item_id)$freq

#単語IDを設定
u_id <- rep(user_id, w)
i_id <- rep(item_id, w)
d_id <- rep(1:d, w)


#インデックスを設定
user_index <- user_ones <- list()
item_index <- item_ones <- list()
for(i in 1:hh){
  user_index[[i]] <- which(user_id==i)
  user_ones[[i]] <- rep(1, length(user_index[[i]]))
}
for(j in 1:item){
  item_index[[j]] <- which(item_id==j)
  item_ones[[j]] <- rep(1, length(item_index[[j]]))
}

##パラメータの設定
#ディリクレ分布の事前分布の設定
alpha11 <- rep(0.2, k11)
alpha12 <- rep(0.2, k12)
alpha21 <- rep(0.15, k21)
alpha22 <- rep(0.15, k22)
alpha3 <- c(0.1, 0.225, 0.3, 0.25, 0.125) * r
alpha41 <- c(rep(0.5, v1/s), rep(0.025, v1/s), rep(0.0025, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha42 <- c(rep(0.3, v1/s), rep(0.1, v1/s), rep(0.025, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha43 <- c(rep(0.2, v1/s), rep(1.0, v1/s), rep(0.2, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha44 <- c(rep(0.025, v1/s), rep(0.1, v1/s), rep(0.3, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha45 <- c(rep(0.0025, v1/s), rep(0.025, v1/s), rep(0.5, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha4 <- rbind(alpha41, alpha42, alpha43, alpha44, alpha45)
alpha51 <- c(rep(0.001, v1/s), rep(0.001, v1/s), rep(0.001, v1/s), rep(0.1, v2), rep(0.002, v3))
alpha52 <- c(rep(0.001, v1/s), rep(0.001, v1/s), rep(0.001, v1/s), rep(0.002, v2), rep(0.1, v3))
beta1 <- c(1.6, 4.8, 5.6)

##すべての単語が出現するまでデータの生成を続ける
for(rp in 1:1000){
  print(rp)
  
  #事前分布からパラメータを生成
  theta11 <- thetat11 <- extraDistr::rdirichlet(hh, alpha11)
  theta12 <- thetat12 <- extraDistr::rdirichlet(item, alpha12)
  theta21 <- thetat21 <- extraDistr::rdirichlet(hh, alpha21)
  theta22 <- thetat22 <- extraDistr::rdirichlet(item, alpha22)
  eta <- etat <- extraDistr::rdirichlet(k11*k12, alpha3)
  omega <- omegat <- extraDistr::rdirichlet(r, alpha4)
  phi <- phit <- extraDistr::rdirichlet(k21, alpha51)
  gamma <- gammat <- extraDistr::rdirichlet(k22, alpha52)
  lambda <- lambdat <- extraDistr::rdirichlet(hh, beta1)
  

  ##モデルに基づきデータを生成
  WX <- matrix(0, nrow=d, ncol=v)
  y <- rep(0, d)
  U1 <- matrix(0, nrow=d, ncol=k11)
  U2 <- matrix(0, nrow=d, ncol=k12)
  Z1_list <- Z21_list <- Z22_list <- wd_list <- list()
  
  for(i in 1:d){
    #ユーザーとアイテムを抽出
    u_index <- user_id[i]
    i_index <- item_id[i]
    
    #評価スコアのトピックを生成
    u1 <- as.numeric(rmnom(1, 1, theta11[u_index, ]))
    u2 <- as.numeric(rmnom(1, 1, theta12[i_index, ]))
    
    #評価スコアのトピックからスコアを生成
    y[i] <- as.numeric(rmnom(1, 1, eta[K1[which.max(u1), which.max(u2)], ]) %*% 1:r)
    K1
    #多項分布からスイッチング変数を生成
    z1 <- rmnom(w[i], 1, lambda[u_index, ])
    z1_vec <- as.numeric(z1 %*% 1:a)
    index_z11 <- which(z1[, 1]==1)
    
    #ユーザートピックを生成
    z21 <- matrix(0, nrow=w[i], ncol=k21)
    index_z21 <- which(z1[, 2]==1)
    if(sum(z1[, 2]) > 0){
      z21[index_z21, ] <- rmnom(sum(z1[, 2]), 1, theta21[u_index, ])
    }
    z21_vec <- as.numeric(z21 %*% 1:k21)
    
    #アイテムトピックを生成
    z22 <- matrix(0, nrow=w[i], ncol=k22)
    index_z22 <- which(z1[, 3]==1)
    if(sum(z1[, 3]) > 0){
      z22[index_z22, ] <- rmnom(sum(z1[, 3]), 1, theta22[i_index, ])
    }
    z22_vec <- as.numeric(z22 %*% 1:k22)
    
    #トピックから単語を生成
    words <- matrix(0, nrow=w[i], ncol=v)
    if(sum(z1[, 1]) > 0){
      words[index_z11, ] <- rmnom(sum(z1[, 1]), 1, omega[y[i], ])
    }
    if(sum(z1[, 2]) > 0){
      words[index_z21, ] <- rmnom(sum(z1[, 2]), 1, phi[z21_vec[index_z21], ])
    }
    if(sum(z1[, 3]) > 0){
      words[index_z22, ] <- rmnom(sum(z1[, 3]), 1, gamma[z22_vec[index_z22], ])
    }
    word_vec <- as.numeric(words %*% 1:v)
    WX[i, ] <- colSums(words)
    
    #データを格納
    wd_list[[i]] <- word_vec
    U1[i, ] <- u1
    U2[i, ] <- u2
    Z1_list[[i]] <- z1
    Z21_list[[i]] <- z21
    Z22_list[[i]] <- z22
  }
  if(min(colSums(WX)) > 0) break
}

#リストを変換
wd <- unlist(wd_list)
Z1 <- do.call(rbind, Z1_list)
Z21 <- do.call(rbind, Z21_list)
Z22 <- do.call(rbind, Z22_list)
storage.mode(Z1) <- "integer"
storage.mode(Z21) <- "integer"
storage.mode(Z22) <- "integer"
storage.mode(WX) <- "integer"


####マルコフ連鎖モンテカルロ法でSwitching Binomial LDAを推定####
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
burnin <- 1000
disp <- 10

##事前分布の設定
alpha11 <- 1.0; alpha12 <- 1.0
alpha21 <- 0.1; alpha22 <- 0.1
alpha31 <- 0.1; alpha32 <- 0.1; alpha33 <- 0.1
beta <- 0.5

##パラメータの真値
theta11 <- thetat11
theta12 <- thetat12
theta21 <- thetat21
theta22 <- thetat22
eta <- etat
phi <- phit
gamma <- gammat
omega <- omegat
lambda <- lambdat

##パラメータの初期値を設定
#トピック分布の初期値
theta11 <- extraDistr::rdirichlet(hh, rep(1.0, k11))
theta12 <- extraDistr::rdirichlet(item, rep(1.0, k12))
theta21 <- extraDistr::rdirichlet(hh, rep(1.0, k21))
theta22 <- extraDistr::rdirichlet(item, rep(1.0, k22))

#単語分布の初期値
eta <- extraDistr::rdirichlet(k11*k12, rep(1.0, r))   #評価スコア分布の初期値
phi <- extraDistr::rdirichlet(k21, rep(1.0, v))   #ユーザーの単語分布の初期値
gamma <- extraDistr::rdirichlet(k22, rep(1.0, v))   #アイテムの単語分布の初期値
omega <- extraDistr::rdirichlet(r, rep(5.0, v))   #評価スコアの単語分布の初期値

#スイッチング変数の初期値
lambda <- matrix(1/s, nrow=hh, ncol=s)


##パラメータの格納用配列
THETA11 <- array(0, dim=c(hh, k11, R/keep))
THETA12 <- array(0, dim=c(item, k12, R/keep))
THETA21 <- array(0, dim=c(hh, k21, R/keep))
THETA22 <- array(0, dim=c(item, k22, R/keep))
ETA <- array(0, dim=c(k11*k12, r, R/keep))
PHI <- array(0, dim=c(k21, v, R/keep))
GAMMA <- array(0, dim=c(k22, v, R/keep))
OMEGA <- array(0, dim=c(r, v, R/keep))
LAMBDA <- array(0, dim=c(hh, s, R/keep))
U_SEG1 <- matrix(0, nrow=d, ncol=k11)
U_SEG2 <- matrix(0, nrow=d, ncol=k12)
SEG1 <- matrix(0, nrow=f, ncol=a)
SEG21 <- matrix(0, nrow=f, ncol=k21)
SEG22 <- matrix(0, nrow=f, ncol=k22)
storage.mode(U_SEG1) <- "integer"
storage.mode(U_SEG2) <- "integer"
storage.mode(SEG21) <- "integer"
storage.mode(SEG22) <- "integer"

##データとインデックスの設定
#インデックスの設定
user_list <- user_vec <- list()
item_list <- item_vec <- list()
wd_list <- wd_vec <- list()
y_list <- y_ones <- list()
user_n <- rep(0, hh)
item_n <- rep(0, item)
for(i in 1:hh){
  user_list[[i]] <- which(u_id==i)
  user_vec[[i]] <- rep(1, length(user_list[[i]]))
  user_n[i] <- sum(user_vec[[i]])
}
for(i in 1:item){
  item_list[[i]] <- which(i_id==i)
  item_vec[[i]] <- rep(1, length(item_list[[i]]))
  item_n[i] <- sum(item_vec[[i]])
}
for(j in 1:v){
  wd_list[[j]] <- which(wd==j)
  wd_vec[[j]] <- rep(1, length(wd_list[[j]]))
}
for(j in 1:r){
  y_list[[j]] <- which(y==j)
  y_ones[[j]] <- rep(1, length(y_list[[j]]))
}
index_k11 <- rep(1:k12, k11)
index_k12 <- rep(1:k11, rep(k12, k11))

#データの設定
y_vec <- y[d_id]
y_data <- matrix(as.numeric(table(1:f, y_vec)), nrow=f, ncol=r)
storage.mode(y_data) <- "integer"
r_vec <- rep(1, r)
a_vec <- rep(1, a)
vec11 <- rep(1, k11)
vec12 <- rep(1, k12)
vec21 <- rep(1, k21)
vec22 <- rep(1, k22)
K11 <- matrix(1:(k11*k12), nrow=k11, ncol=k12, byrow=T)
K12 <- matrix(1:(k11*k12), nrow=k12, ncol=k11, byrow=T)


##対数尤度の基準値
par <- colSums(WX) / f
LLst <- sum(WX %*% log(par))


####ギブスサンプリングでパラメータをサンプリング####
for(rp in 1:R){
  
  ##評点スコアのユーザートピックをサンプリング
  #ユーザートピックの条件付き確率
  eta11 <- t(eta)[y, ] * theta12[item_id, index_k11]
  par_u0 <- matrix(0, nrow=d, ncol=k11)
  for(j in 1:k11){
    par_u0[, j] <- eta11[, K1[j, ]] %*% vec11
  }
  par_u1 <- theta11[user_id, ] * par_u0   #ユーザートピックの期待尤度
  
  #潜在変数の割当確率からトピックをサンプリング
  u1_rate <- par_u1 / as.numeric(par_u1 %*% vec11)
  Ui1 <- rmnom(d, 1, u1_rate)
  Ui1_T <- t(Ui1)
  
  
  ##評点スコアのアイテムトピックをサンプリング
  #アイテムトピックの条件付き確率
  eta12 <- t(eta)[y, ] * theta11[item_id, index_k12]
  par_u0 <- matrix(0, nrow=d, ncol=k12)
  for(j in 1:k12){
    par_u0[, j] <- eta12[, K1[, j]] %*% vec12
  }
  par_u2 <- theta12[item_id, ] * par_u0   #アイテムトピックの期待尤度
  
  #潜在変数の割当確率からトピックをサンプリング
  u2_rate <- par_u2 / as.numeric(par_u2 %*% vec12)
  Ui2 <- rmnom(d, 1, u2_rate)
  Ui2_T <- t(Ui2)

  #ユーザーとアイテムトピックを統合
  Ui <- Ui2[, index_k11] * Ui1[, index_k12]
  Ui_T <- t(Ui)
  
  
  ##評点スコアのユーザーおよびアイテムのトピックをサンプリング
  #ユーザーのトピック分布をサンプリング
  wusum0 <- matrix(0, nrow=d, ncol=k11)
  for(i in 1:hh){
    wusum0[i, ] <- Ui1_T[, user_index[[i]]] %*% user_ones[[i]]
  }
  wusum <- wusum0 + alpha11   #ディリクレ分布のパラメータ
  theta11 <- extraDistr::rdirichlet(hh, wusum)   #ディリクレ分布からtheta11をサンプリング
  
  #アイテムのトピック分布をサンプリング
  wisum0 <- matrix(0, nrow=item, ncol=k12)
  for(i in 1:item){
    wisum0[i, ] <- Ui2_T[, item_index[[i]]] %*% item_ones[[i]]
  }
  wisum <- wisum0 + alpha12   #ディリクレ分布のパラメータ
  theta12 <- extraDistr::rdirichlet(item, wisum)   #ディリクレ分布からtheta11をサンプリング
  
  
  ##評価スコア分布をサンプリング
  vsum0 <- matrix(0, nrow=k11*k12, ncol=r)
  for(j in 1:r){
    vsum0[, j] <- Ui_T[, y_list[[j]]] %*% y_ones[[j]]
  }
  vsum <- vsum0 + alpha31   #ディリクレ分布のパラメータ
  eta <- extraDistr::rdirichlet(k11*k12, vsum)   #ディリクレ分布からetaをサンプリング
  
  
  ##多項分布よりスイッチング変数をサンプリング
  #評価スコア、ユーザーおよびアイテムの期待尤度を設定
  Li_score <- as.numeric((t(omega)[wd, ] * y_data) %*% r_vec)   #スコア尤度
  Li_user <- theta21[u_id, ] * t(phi)[wd, ]   #ユーザー尤度
  par_user <- as.numeric(Li_user %*% vec21)   #ユーザーの期待尤度
  Li_item <- theta22[i_id, ] * t(gamma)[wd, ]   #アイテム尤度
  par_item <- as.numeric(Li_item %*% vec22)   #アイテムの期待尤度
  par <- cbind(Li_score, par_user, par_item)
  
  #潜在確率からスイッチング変数を生成
  lambda_r <- lambda[u_id, ]   #スイッチング変数の事前分布
  par_r <- lambda_r * par
  s_prob <- par_r / as.numeric(par_r %*% a_vec)   #スイッチング変数の割当確率
  Zi1 <- rmnom(f, 1, s_prob)   #多項分布からスイッチング変数を生成
  Zi1_T <- t(Zi1)
  index_z21 <- which(Zi1[, 2]==1)
  index_z22 <- which(Zi1[, 3]==1)
  
  #ディリクレ分布から混合率をサンプリング
  rsum0 <- matrix(0, nrow=hh, ncol=a)
  for(i in 1:hh){
    rsum0[i, ] <- Zi1_T[, user_list[[i]]] %*% user_vec[[i]]
  }
  rsum <- rsum0 + beta   #ディリクレ分布のパラメータ
  lambda <- extraDistr::rdirichlet(hh, rsum)   #ディリクレ分布からlambdaをサンプリング
  
  
  ##ユーザーおよびアイテムのトピックをサンプリング
  #トピックの割当確率を推定
  z_rate1 <- Li_user[index_z21, ] / par_user[index_z21]   #ユーザーのトピック割当確率
  z_rate2 <- Li_item[index_z22, ] / par_item[index_z22]   #アイテムのトピック割当確率
  
  #多項分布からトピックを生成
  Zi21 <- matrix(0, nrow=f, ncol=k21)
  Zi22 <- matrix(0, nrow=f, ncol=k22)
  Zi21[index_z21, ] <- rmnom(nrow(z_rate1), 1, z_rate1)
  Zi22[index_z22, ] <- rmnom(nrow(z_rate2), 1, z_rate2)
  Zi21_T <- t(Zi21)
  Zi22_T <- t(Zi22)
  
  
  ##トピックモデルのパラメータをサンプリング
  #ユーザーのトピック分布をサンプリング
  wusum0 <- matrix(0, nrow=hh, ncol=k21)
  for(i in 1:hh){
    wusum0[i, ] <- Zi21_T[, user_list[[i]]] %*% user_vec[[i]]
  }
  wusum <- wusum0 + alpha21   #ディリクレ分布のパラメータ
  theta21 <- extraDistr::rdirichlet(hh, wusum)   #ディリクレ分布からtheta21をサンプリング
  
  #アイテムのトピック分布をサンプリング
  wisum0 <- matrix(0, nrow=item, ncol=k22)
  for(i in 1:item){
    wisum0[i, ] <- Zi22_T[, item_list[[i]]] %*% item_vec[[i]]
  }
  wisum <- wisum0 + alpha22   #ディリクレ分布のパラメータ
  theta22 <- extraDistr::rdirichlet(item, wisum)   #ディリクレ分布からtheta22をサンプリング
  
  
  ##評価スコア、ユーザーおよびアイテムの単語分布をサンプリング
  y_data_t <- t(y_data * Zi1[, 1])
  vssum0 <- matrix(0, nrow=r, ncol=v)
  vusum0 <- matrix(0, nrow=k21, ncol=v)
  visum0 <- matrix(0, nrow=k22, ncol=v)
  for(j in 1:v){
    vssum0[, j] <- y_data_t[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]
    vusum0[, j] <- Zi21_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]
    visum0[, j] <- Zi22_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]
  }
  vssum <- vssum0 + alpha31; vusum <- vusum0 + alpha32; visum <- visum0 + alpha33
  omega <- extraDistr::rdirichlet(r, vssum)
  phi <- extraDistr::rdirichlet(k21, vusum)
  gamma <- extraDistr::rdirichlet(k22, visum)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA11[, , mkeep] <- theta11
    THETA12[, , mkeep] <- theta12
    THETA21[, , mkeep] <- theta21
    THETA22[, , mkeep] <- theta22
    ETA[, , mkeep] <- eta
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma
    OMEGA[, , mkeep] <- omega
    LAMBDA[, , mkeep] <- lambda
  }  
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    U_SEG1 <- U_SEG1 + Ui1
    U_SEG2 <- U_SEG2 + Ui2
    SEG1 <- SEG1 + Zi1
    SEG21 <- SEG21 + Zi21
    SEG22 <- SEG22 + Zi22
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL1 <- sum(log(rowSums(par_u1)))
    LL2 <- sum(log(rowSums(Zi1 * par)))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL1, LL2, LLst))
    print(round(c(colMeans(Zi1), colMeans(Z1)), 3))
    print(round(cbind(phi[, (v1-5):(v1+4)], phit[, (v1-5):(v1+4)]), 3))
  }
}

####サンプリング結果の可視化と要約####
burnin <- 1000/keep
RS <- R/keep

##サンプリング結果のプロット
#スイッチング変数の混合率の可視化
matplot(t(LAMBDA[, 1, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(LAMBDA[, 2, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(LAMBDA[, 3, ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#評価スコアのトピック分布のサンプリング結果をプロット
matplot(t(THETA11[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA11[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA11[250, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA11[500, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA11[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[50, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[150, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA12[200, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#評価スコアのトピック分布のサンプリング結果をプロット
matplot(t(THETA21[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[250, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[500, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA21[1000, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[50, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[100, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[150, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(THETA22[200, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#スコア分布のサンプリング結果の可視化
matplot(t(ETA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[10, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[15, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(ETA[20, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

#単語分布のサンプリング結果の可視化
matplot(t(PHI[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[7, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(PHI[9, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[4, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[8, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[12, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(GAMMA[15, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[1, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[2, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[3, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[4, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")
matplot(t(OMEGA[5, , ]), type="l", xlab="サンプリング回数", ylab="パラメータ")

##サンプリング結果の要約
#サンプリング結果の事後平均
round(cbind(apply(LAMBDA[, , burnin:RS], c(1, 2), mean), lambdat), 3)   #スイッチング変数の混合率の事後平均
round(cbind(apply(THETA11[, , burnin:RS], c(1, 2), mean), thetat11), 3)   #ユーザーのトピック割当の事後平均
round(cbind(apply(THETA12[, , burnin:RS], c(1, 2), mean), thetat12), 3)   #アイテムのトピック割当の事後平均
round(cbind(t(apply(PHI[, , burnin:RS], c(1, 2), mean)), t(phit)), 3)   #ユーザーの単語分布の事後平均
round(cbind(t(apply(GAMMA[, , burnin:RS], c(1, 2), mean)), t(gammat)), 3)   #アイテムの単語分布の事後平均
round(cbind(t(apply(OMEGA[, , burnin:RS], c(1, 2), mean)), t(omegat)), 3)   #評価スコアの単語分布の事後平均

#サンプリング結果の事後信用区間
round(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(LAMBDA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(apply(THETA1[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025)), 3)
round(apply(THETA2[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975)), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(PHI[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)
round(t(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(GAMMA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)
round(t(apply(OMEGA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.025))), 3)
round(t(apply(OMEGA[, , burnin:RS], c(1, 2), function(x) quantile(x, 0.975))), 3)


##サンプリングされた潜在変数の要約
n <- max(SEG1)
round(cbind(SEG1/n, Z1), 3)
round(cbind(rowSums(SEG21), SEG21/max(rowSums(SEG21)), Z21 %*% 1:k21), 3)
round(cbind(rowSums(SEG22), SEG22/max(rowSums(SEG22)), Z22 %*% 1:k22), 3)
