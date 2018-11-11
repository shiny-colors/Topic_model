#####結合トピックモデル#####
library(MASS)
library(lda)
library(RMeCab)
library(gtools)
library(reshape2)
library(dplyr)
library(plyr)
library(ggplot2)


####データの発生####
#set.seed(423943)
#データの設定
k <- 5   #トピック数
d <- 200   #文書数
v <- 100   #語彙数
w <- 200   #1文書あたりの単語数 
m <- 20   #タグ数
g <- 10   #1文書あたりのタグ数

#パラメータの設定
alpha0 <- runif(k, 0.1, 0.8)   #文書のディレクリ事前分布のパラメータ
alpha1 <- rep(0.25, v)   #単語のディレクリ事前分布のパラメータ
alpha2 <- rep(0.15, m)   #タグのディレクリ事前分布のパラメータ

#ディレクリ乱数の発生
theta0 <- rdirichlet(d, alpha0)   #文書のトピック分布をディレクリ乱数から発生
phi0 <- rdirichlet(k, alpha1)   #単語のトピック分布をディレクリ乱数から発生
gamma0 <- rdirichlet(k, alpha2)

#多項分布の乱数からデータを発生
WX <- matrix(0, d, v)
TX <- matrix(0, d, m)
ZS1 <- list()
ZS2 <- list()

for(i in 1:d){
  #トピックを生成
  z1 <- t(rmultinom(w, 1, theta0[i, ]))   #文書の単語のトピックを生成
  z2 <- t(rmultinom(g, 1, theta0[i, ]))   #文書のタグのトピックを生成
  zn1 <- z1 %*% c(1:k)   #0,1を数値に置き換える
  zdn1 <- cbind(zn1, z1)   #apply関数で使えるように行列にしておく
  zn2 <- z2 %*% c(1:k)   
  zdn2 <- cbind(zn2, z2)

  #トピックから応答変数を生成
  wn <- t(apply(zdn1, 1, function(x) rmultinom(1, 1, phi0[x[1], ])))   #文書のトピックから単語を生成
  tn <- t(apply(zdn2, 1, function(x) rmultinom(1, 1, gamma0[x[1], ])))   #文書のトピックからタグを生成
  
  wdn <- colSums(wn)   #単語ごとに合計して1行にまとめる
  tdn <- colSums(tn)   #タグごとに合計して1行にまとめる 
  WX[i, ] <- wdn  
  TX[i, ] <- tdn
  ZS1[[i]] <- cbind(rep(i, w), zdn1[, 1])
  ZS2[[i]] <- cbind(rep(i, g), zdn2[, 1])
  print(i)
}


#リストを行列方式に変更
ZS1 <- do.call(rbind, ZS1)
ZS2 <- do.call(rbind, ZS2)

#トピックの単純集計
z1_table <- table(ZS1[, 2])
z2_table <- table(ZS2[, 2])
z1_r <- z1_table/sum(z1_table)
z2_r <- z2_table/sum(z2_table)

barplot(z1_table, names.arg=c("seg1", "seg2", "seg3", "seg4", "seg5"))
barplot(z2_table, names.arg=c("seg1", "seg2", "seg3", "seg4", "seg5"))

round(colSums(WX)/sum(WX), 3)   #単語の出現頻度
round(colSums(TX)/sum(TX), 3)   #タグの出現頻度


####マルコフ連鎖モンテカルロ法で結合トピックモデルを推定####
####マルコフ連鎖モンテカルロ法の設定####
R <- 10000   #サンプリング回数
keep <- 2   
iter <- 0

#ハイパーパラメータの事前分布の設定
alpha <- alpha0   #文書のディレクリ事前分布のパラメータ
beta <- alpha1[1]   #単語のディレクリ事前分布のパラメータ
gamma <- alpha2[2]   #タグのディレクリ事前分布のパラメータ

#パラメータの初期値
theta.ini <- runif(k, 0.3, 1.5)
phi.ini <- runif(v, 0.5, 1)
psi.ini <- runif(m, 0.5, 1)
theta <- rdirichlet(d, theta.ini)   #文書トピックのパラメータの初期値
phi <- rdirichlet(k, phi.ini)   #単語トピックのパラメータの初期値
psi <- rdirichlet(k, psi.ini)   #タグトピックのパラメータの初期値

#パラメータの格納用配列
THETA <- array(0, dim=c(d, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
PSI <- array(0, dim=c(k, m, R/keep))
W.SEG <- matrix(0, nrow=d*w, R/keep)
T.SEG <- matrix(0, nrow=d*g, R/keep)
RATE1 <- matrix(0, nrow=R/keep, ncol=k)
RATE2 <- matrix(0, nrow=R/keep, ncol=k)


####データの準備####
#IDを作成
d.id1 <- rep(1:d, rep(v, d))
w.id <- rep(1:v, d) 
d.id2 <- rep(1:d, rep(m, d))
g.id <- rep(1:m, d)
ID1 <- data.frame(d.id=d.id1, w.id=w.id)
ID2 <- data.frame(d.id=d.id2, g.id=g.id)

#インデックスを作成
index_w <- matrix(1:nrow(ID1), nrow=d, ncol=v, byrow=T)
index_g <- matrix(1:nrow(ID2), nrow=d, ncol=m, byrow=T)

index_g
cbind(X1_Z, as.numeric(t(WX)))
cbind(X2_Z, as.numeric(t(TX)))

##トピック割当の初期値を生成
#トピック割当の格納用
X1_Z <- matrix(0, nrow=nrow(ID1), ncol=k)
X2_Z <- matrix(0, nrow=nrow(ID2), ncol=k)

#文書ごとに単語およびタグのトピックを生成
for(i in 1:d){
  
  #thetaを行列形式に変更
  theta.m1 <- matrix(theta[i, ], nrow=k, ncol=v) 
  theta.m2 <- matrix(theta[i, ], nrow=k, ncol=m)
  
  #混合率を計算
  z1.rate <- t(phi * theta.m1) / matrix(rowSums(t(phi * theta.m1)), nrow=v, ncol=k)
  z2.rate <- t(psi * theta.m2) / matrix(rowSums(t(psi * theta.m2)), nrow=m, ncol=k)
  
  #単語とタグのトピック割当を決定
  X1_Z[index_w[i, ], ] <- t(apply(cbind(WX[i, ], z1.rate), 1, function(x) rmultinom(1, x[1], x[-1])))
  X2_Z[index_g[i, ], ] <- t(apply(cbind(TX[i, ], z2.rate), 1, function(x) rmultinom(1, x[1], x[-1])))
}

##トピック割当数の初期値を計算
#全体でのトピック割当
k1_sum <- colSums(X1_Z)
k2_sum <- colSums(X2_Z)

#文書ごとのトピック割当
kw_sum <- matrix(0, nrow=d, ncol=k)
kg_sum <- matrix(0, nrow=d, ncol=k)

for(i in 1:d){
  kw_sum[i, ] <- colSums(X1_Z[index_w[i, ], ])
  kg_sum[i, ] <- colSums(X2_Z[index_g[i, ], ])
}

#単語およびタグごとのトピック割当
#単語ごとの割当
kv_sum <- matrix(0, nrow=v, ncol=k)
for(i in 1:v){ kv_sum[i, ] <- colSums(X1_Z[index_w[, i], ])}

#タグごとの割当
km_sum <- matrix(0, nrow=m, ncol=k)
for(i in 1:m){ km_sum[i, ] <- colSums(X2_Z[index_g[, i], ])}

#トピック割当をベクトル形式に変更
seg_vec1 <- unlist(apply(X1_Z, 1, function(x) rep(1:k, x)))
seg_vec2 <- unlist(apply(X2_Z, 1, function(x) rep(1:k, x)))

#行列形式に変更
seg_mx1 <- matrix(0, nrow=length(seg_vec1), ncol=k)
seg_mx2 <- matrix(0, nrow=length(seg_vec2), ncol=k)
for(i in 1:nrow(seg_mx1)) {seg_mx1[i, seg_vec1[i]] <- 1}
for(i in 1:nrow(seg_mx2)) {seg_mx2[i, seg_vec2[i]] <- 1}


##トピック割当ベクトルのIDを作成
id_vec11 <- rep(1:d, rep(w, d))
id_vec21 <- rep(1:d, rep(g, d))
id_vec12 <- c()
id_vec22 <- c()

for(i in 1:d){
  id_vec12 <- c(id_vec12, rep(1:v, rowSums(X1_Z[index_w[i, ], ])))
  id_vec22 <- c(id_vec22, rep(1:m, rowSums(X2_Z[index_g[i, ], ])))
}

Z1 <- matrix(0, nrow=d*w, k)
Z2 <- matrix(0, nrow=d*g, k)

#ギブスサンプリング用のインデックスを作成
index_word <- list()
index_tag <- list()

for(i in 1:d){
  index_word[[i]] <- subset(1:length(id_vec11), id_vec11==i)
  index_tag[[i]] <- subset(1:length(id_vec21), id_vec21==i)
}


####周辺化ギブスサンプリングで推定####
for(rp in 1:R){
  
  ##トピックをサンプリング
  for(i in 1:d){
    ##単語のトピックをサンプリング
    for(wd in 1:length(index_word[[i]])){
      index1 <- index_word[[i]][wd]   #単語のインデックス
      
      #トピック生成する単語のトピックを取り除く
      mx1 <- seg_mx1[index1, ]
      k1 <- k1_sum - mx1
      kw <- kw_sum[i, ] - mx1
      kv <- kv_sum[id_vec12[index1], ] - mx1
      
      #単語のトピック割当確率を計算
      z1_sums <- (kw + kg_sum[i, ] + alpha) * (kv + beta) / (k1 + beta*v)
      z1_rate <- z1_sums / sum(z1_sums)
      
      #トピックをサンプリング
      Z1 <- t(rmultinom(1, 1, z1_rate))
      
      #データを更新
      k1_sum <- k1 + Z1
      kw_sum[i, ] <- kw + Z1
      kv_sum[id_vec12[index1], ] <- kv + Z1
      seg_mx1[index1, ] <- Z1
    }
    
    ##タグのトピックをサンプリング
    for(tg in 1:length(index_tag[[i]])){
      index2 <- index_tag[[i]][tg]   #タグのインデックス
      
      #トピック生成する単語のトピックを取り除く
      mx2 <- seg_mx2[index2, ]
      k2 <- k2_sum - mx2
      kg <- kg_sum[i, ] - mx2
      km <- km_sum[id_vec22[index2], ] - mx2

      #タグのトピック割当確率を計算
      z2_sums <- (kg + kw_sum[i, ] + alpha) * (km + gamma) / (k2 + gamma*m)
      z2_rate <- z2_sums / sum(z2_sums)
      
      #トピックをサンプリング
      Z2 <- t(rmultinom(1, 1, z2_sums))
      
      #データを更新
      k2_sum <- k2 + Z2
      kg_sum[i, ] <- kg + Z2
      km_sum[id_vec22[index2], ] <- km + Z2
      seg_mx2[index2, ] <- Z2
    }
  }
  
  ##サンプリング結果を保存
  mkeep <- rp/keep
  if(rp%%keep==0){
    
    #混合率の計算
    rate11 <- colSums(seg_mx1)/nrow(seg_mx1)
    rate21 <- colSums(seg_mx2)/nrow(seg_mx2)
    
    #サンプリング結果を保存
    W.SEG[, mkeep] <- seg_mx1 %*% 1:k 
    T.SEG[, mkeep] <- seg_mx2 %*% 1:k
    RATE1[mkeep, ] <- rate11
    RATE2[mkeep, ] <- rate21
    
    #サンプリング状況を表示
    print(rp)
    print(round(rbind(rate11, rate12=z1_r), 3))
    print(round(rbind(rate21, rate22=z2_r), 3))
  }
}

####推定結果と集計####
burnin <- 1000   #バーンイン期間は2000回まで

#サンプリング結果の可視化
matplot(RATE1, type="l", ylab="混合率", main="単語トピックの混合率")
matplot(RATE2, type="l", ylab="混合率", main="タグトピックの混合率")

#混合率の事後平均
round(rbind(rate1_mcmc=colMeans(RATE1[burnin:nrow(RATE1), ]), rate1_true=z1_r), 3)
round(rbind(rate2_mcmc=colMeans(RATE2[burnin:nrow(RATE2), ]), rate2_true=z2_r), 3)

##推定されたトピック分布
w.seg_freq <- matrix(0, nrow=nrow(W.SEG), ncol=k)
t.seg_freq <- matrix(0, nrow=nrow(T.SEG), ncol=k)

#単語ごとにトピック分布を計算
for(i in 1:nrow(W.SEG)){
  print(i)
  w.seg_freq[i, ] <- table(c(W.SEG[i, (burnin+1):(R/keep)], 1:k)) - rep(1, k)
}
w.seg_rate <- w.seg_freq / length((burnin+1):(R/keep))

#タグごとにトピック分布を計算
for(i in 1:nrow(T.SEG)){
  print(i)
  t.seg_freq[i, ] <- table(c(T.SEG[i, (burnin+1):(R/keep)], 1:k)) - rep(1, k)
}
t.seg_rate <- t.seg_freq / length((burnin+1):(R/keep))

##トピック分布からパラメータを推定
theta_w <- matrix(0, nrow=d, ncol=k)
theta_t <- matrix(0, nrow=d, ncol=k)

#文書ごとにトピック分布のパラメータを計算
for(i in 1:d){
 theta_w[i, ]  <- colSums(w.seg_rate[id_vec11==i, ])/w
 theta_t[i, ]  <- colSums(t.seg_rate[id_vec21==i, ])/g
}

#トピック分布のパラメータと真のパラメータの比較
round(data.frame(word=theta_w, tag=theta_t, topic=theta0), 3)



