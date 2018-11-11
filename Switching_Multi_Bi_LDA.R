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
k21 <- 15   #ユーザーのテキストのトピック数
k22 <- 15   #アイテムのテキストのトピック数
hh <- 3000   #レビュアー数
item <- 1500   #アイテム数
v1 <- 300   #評価スコアの語彙数
v2 <- 400   #ユーザートピックの語彙数
v3 <- 400   #アイテムトピックの語彙数
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
repeat { 
  m_vec <- rep(0, hh*item)
  for(i in 1:item){
    prob <- runif(1, 0.005, 0.07)
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
user_index <- list()
item_index <- list()
for(i in 1:hh){
  user_index[[i]] <- which(user_id==i)
}
for(j in 1:item){
  item_index[[j]] <- which(item_id==j)
}

##パラメータの設定
#トピック分布の事前分布の設定
alpha11 <- rep(0.2, k11); alpha12 <- rep(0.2, k12)
alpha21 <- rep(0.15, k21); alpha22 <- rep(0.15, k22)
alpha3 <- c(0.1, 0.225, 0.3, 0.25, 0.125) * r
beta1 <- c(1.6, 4.8, 5.6)   #スイッチング変数の事前分布

#評価スコアの単語分布の事前分布の設定
alpha41 <- c(rep(0.5, v1/s), rep(0.025, v1/s), rep(0.0025, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha42 <- c(rep(0.3, v1/s), rep(0.1, v1/s), rep(0.025, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha43 <- c(rep(0.2, v1/s), rep(1.0, v1/s), rep(0.2, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha44 <- c(rep(0.025, v1/s), rep(0.1, v1/s), rep(0.3, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha45 <- c(rep(0.0025, v1/s), rep(0.025, v1/s), rep(0.5, v1/s), rep(0.001, v2), rep(0.001, v3))
alpha4 <- rbind(alpha41, alpha42, alpha43, alpha44, alpha45)

#ユーザーとアイテムの単語分布の事前分布の設定
alpha51 <- c(rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.05, v2), rep(0.0001, v3))
alpha52 <- c(rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.0001, v1/s), rep(0.001, v2), rep(0.05, v3))


##すべての単語が出現するまでデータの生成を続ける
rp <- 0 
repeat {
  rp <- rp + 1
  print(rp)
  
  #トピック分布のパラメータを生成
  theta11 <- thetat11 <- extraDistr::rdirichlet(hh, alpha11)
  theta12 <- thetat12 <- extraDistr::rdirichlet(item, alpha12)
  theta21 <- thetat21 <- extraDistr::rdirichlet(hh, alpha21)
  theta22 <- thetat22 <- extraDistr::rdirichlet(item, alpha22)
  eta <- etat <- extraDistr::rdirichlet(k11*k12, alpha3)
  lambda <- lambdat <- extraDistr::rdirichlet(hh, beta1)   #スイッチング変数のパラメータ
  
  #単語分布のパラメータを生成
  omega <- omegat <- extraDistr::rdirichlet(r, alpha4)
  phi <- phit <- extraDistr::rdirichlet(k21, alpha51)
  gamma <- gammat <- extraDistr::rdirichlet(k22, alpha52)
  
  #出現確率が低い単語を入れ替える
  index <- which(colMaxs(phi[, alpha51==max(alpha51)]) < (k21*k21)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1, k21))) %*% 1:k21), index[j]] <- (k21*k21)/f
  }
  index <- which(colMaxs(gamma[, alpha52==max(alpha52)]) < (k22*k22)/f)
  for(j in 1:length(index)){
    gamma[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(1, k22))) %*% 1:k22), index[j]] <- (k22*k22)/f
  }
  
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

#スパース行列に変換
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
word_dt <- t(word_data)

#オブジェクトを消去
rm(Z1_list); rm(Z21_list); rm(Z22_list)
rm(WX); rm(wd_list)
gc(); gc()


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
R <- 2000
keep <- 2  
iter <- 0
burnin <- 300/keep
disp <- 10

##事前分布の設定
#トピック分布とスイッチング変数の事前分布
alpha11 <- 0.25; alpha12 <- 0.25
alpha21 <- 0.1; alpha22 <- 0.1
beta <- 0.5

#単語分布の事前分布
alpha31 <- alpha32 <- alpha33 <- 0.1


##パラメータの真値
#トピック分布とスイッチング変数の真値
theta11 <- thetat11
theta12 <- thetat12
theta21 <- thetat21
theta22 <- thetat22
lambda <- lambdat

#スコア分布と単語分布の真値
eta <- etat
phi <- phit
gamma <- gammat
omega <- omegat


##パラメータの初期値を設定
#トピック分布とスイッチング変数の初期値
theta11 <- extraDistr::rdirichlet(hh, rep(1.0, k11))
theta12 <- extraDistr::rdirichlet(item, rep(1.0, k12))
theta21 <- extraDistr::rdirichlet(hh, rep(1.0, k21))
theta22 <- extraDistr::rdirichlet(item, rep(1.0, k22))
lambda <- matrix(1/s, nrow=hh, ncol=s)

#スコア分布と単語分布の初期値
eta <- extraDistr::rdirichlet(k11*k12, rep(1.0, r))   #評価スコア分布の初期値
phi <- extraDistr::rdirichlet(k21, rep(2.0, v))   #ユーザーの単語分布の初期値
gamma <- extraDistr::rdirichlet(k22, rep(2.0, v))   #アイテムの単語分布の初期値
omega <- extraDistr::rdirichlet(r, rep(2.0, v))   #評価スコアの単語分布の初期値


##パラメータの格納用配列
#トピック分布とスイッチング変数の格納用配列
THETA11 <- array(0, dim=c(hh, k11, R/keep))
THETA12 <- array(0, dim=c(item, k12, R/keep))
THETA21 <- array(0, dim=c(hh, k21, R/keep))
THETA22 <- array(0, dim=c(item, k22, R/keep))
LAMBDA <- array(0, dim=c(hh, s, R/keep))

#スコア分布と単語分布の格納用配列
ETA <- array(0, dim=c(k11*k12, r, R/keep))
PHI <- array(0, dim=c(k21, v, R/keep))
GAMMA <- array(0, dim=c(k22, v, R/keep))
OMEGA <- array(0, dim=c(r, v, R/keep))

#トピックの格納用配列
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
#ユーザーのインデックス
user_dt <- sparseMatrix(u_id, 1:f, x=rep(1, f), dims=c(hh, f))
user_n <- rowSums(user_dt)

#アイテムのインデックス
item_dt <- sparseMatrix(i_id, 1:f, x=rep(1, f), dims=c(item, f))
item_n <- rowSums(item_dt)

#単語のインデックス
wd_dt <- t(word_data)
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
#ユニグラムモデルの対数尤度
LLst <- sum(word_data %*% log(colSums(word_data) / f))

#真値での対数尤度
Li_score <- as.numeric((t(omegat)[wd, ] * y_data) %*% r_vec)   #スコア尤度
Li_user <- thetat21[u_id, ] * t(phit)[wd, ]; par_user <- as.numeric(Li_user %*% vec21)   #ユーザー尤度
Li_item <- thetat22[i_id, ] * t(gammat)[wd, ]; par_item <- as.numeric(Li_item %*% vec22)   #アイテム尤度
par <- rowSums(Z1 * cbind(Li_score, par_user, par_item))   #期待尤度
LLbest <- sum(log(par[which(par!=0)]))   #対数尤度の和
gc(); gc()


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
  
  
  ##多項分布よりスイッチング変数をサンプリング
  #評価スコア、ユーザーおよびアイテムの期待尤度を設定
  Li_score <- as.numeric((t(omega)[wd, ] * y_data) %*% r_vec)   #スコア尤度
  Li_user <- theta21[u_id, ] * t(phi)[wd, ]; par_user <- as.numeric(Li_user %*% vec21)   #ユーザー尤度
  Li_item <- theta22[i_id, ] * t(gamma)[wd, ]; par_item <- as.numeric(Li_item %*% vec22)   #アイテム尤度
  par <- cbind(Li_score, par_user, par_item)   #期待尤度
  
  #潜在確率からスイッチング変数を生成
  lambda_r <- lambda[u_id, ]   #スイッチング変数の事前分布
  par_r <- lambda_r * par
  s_prob <- par_r / as.numeric(par_r %*% a_vec)   #スイッチング変数の割当確率
  Zi1 <- rmnom(f, 1, s_prob)   #多項分布からスイッチング変数を生成
  index_z21 <- which(Zi1[, 2]==1)
  index_z22 <- which(Zi1[, 3]==1)
  
  #ディリクレ分布から混合率をサンプリング
  rsum <- as.matrix(user_dt %*% Zi1) + beta   #ディリクリ分布のパラメータ
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

  
  ##トピックモデルのパラメータをサンプリング
  #ユーザーのトピック分布をサンプリング
  wusum <- as.matrix(user_dt %*% Zi21) + alpha21   #ディリクレ分布のパラメータ
  theta21 <- extraDistr::rdirichlet(hh, wusum)   #パラメータをサンプリング
  
  #アイテムのトピック分布をサンプリング
  wisum <- as.matrix(item_dt %*% Zi22) + alpha22   #ディリクレ分布のパラメータ
  theta22 <- extraDistr::rdirichlet(item, wisum)   #パラメータをサンプリング
  
  
  ##評価スコア、ユーザーおよびアイテムの単語分布をサンプリング
  #ディリクレ分布のパラメータ
  Zi1_y <- y_data * Zi1[, 1]
  vssum <- t(as.matrix(wd_dt %*% Zi1_y)) + alpha31
  vusum <- t(as.matrix(wd_dt %*% Zi21)) + alpha32
  visum <- t(as.matrix(wd_dt %*% Zi22)) + alpha33
  
  #ディクリ分布からパラメータをサンプリング
  omega <- extraDistr::rdirichlet(r, vssum)
  phi <- extraDistr::rdirichlet(k21, vusum)
  gamma <- extraDistr::rdirichlet(k22, visum)
  
  
  ##パラメータの格納とサンプリング結果の表示
  #サンプリングされたパラメータを格納
  if(rp%%keep==0){
    #サンプリング結果の格納
    mkeep <- rp/keep
    THETA21[, , mkeep] <- theta21
    THETA22[, , mkeep] <- theta22
    PHI[, , mkeep] <- phi
    GAMMA[, , mkeep] <- gamma
    OMEGA[, , mkeep] <- omega
    LAMBDA[, , mkeep] <- lambda
  }  
  
  #トピック割当はバーンイン期間を超えたら格納する
  if(rp%%keep==0 & rp >= burnin){
    SEG1 <- SEG1 + Zi1
    SEG21 <- SEG21 + Zi21
    SEG22 <- SEG22 + Zi22
  }
  
  if(rp%%disp==0){
    #対数尤度を計算
    LL <- sum(log(rowSums(Zi1 * par)))
    
    #サンプリング結果を確認
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(c(colMeans(Zi1), colMeans(Z1)), 3))
    print(round(cbind(phi[, (v1-5):(v1+4)], phit[, (v1-5):(v1+4)]), 3))
  }
}

