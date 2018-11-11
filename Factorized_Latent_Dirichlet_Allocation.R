#####Factorized Latent Dirichlet Allocation#####
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

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
k <- 15   #�g�s�b�N��
hh <- 5000   #���[�U�[��
item <- 2500   #�A�C�e����
v <- 1000   #��b�� 
w <- rpois(item, rgamma(item, 60, 0.4))   #1����������̌�b��
pt <- rtpois(hh, rgamma(hh, 25.0, 0.225), a=1, b=Inf)   #�w���ڐG��
f <- sum(w)   #����b��
hhpt <- sum(pt)   #���X�R�A��
vec_k <- rep(1, k)

#ID�̐ݒ�
d_id <- rep(1:item, w)   #����ID
no_id <- as.numeric(unlist(tapply(1:f, d_id, rank)))
user_id <- rep(1:hh, pt)   #���[�U�[ID
t_id <- as.numeric(unlist(tapply(1:hhpt, user_id, rank)))
user_list <- list()
for(i in 1:hh){
  user_list[[i]] <- which(user_id==i)
}

##�A�C�e���̊����𐶐�
#�Z�O�����g�����𐶐�
topic <- 25
phi <- extraDistr::rdirichlet(topic, rep(0.5, item))
z <- as.numeric(rmnom(hh, 1,  extraDistr::rdirichlet(hh, rep(2.5, topic))) %*% 1:topic)

#�������z����A�C�e���𐶐�
item_id_list <- list()
for(i in 1:hh){
  if(i%%100==0){
    print(i)
  }
  item_id_list[[i]] <- as.numeric(rmnom(pt[i], 1, phi[z[user_id[user_list[[i]]]], ]) %*% 1:item)
}
item_id <- unlist(item_id_list)
item_list <- list(); item_n <- rep(0, item)
for(j in 1:item){
  item_list[[j]] <- which(item_id==j)
  item_n[j] <- length(item_list[[j]])   
}

#�X�p�[�X�s����쐬
user_data <- sparseMatrix(1:hhpt, user_id, x=rep(1, hhpt), dims=c(hhpt, hh))
user_data_T <- t(user_data)
item_data <- sparseMatrix(1:hhpt, item_id, x=rep(1, hhpt), dims=c(hhpt, item))
item_data_T <- t(item_data)

#���������f�[�^������
freq_item <- plyr::count(item_id); freq_item$x <- as.character(freq_item$x)
hist(freq_item$freq, breaks=25, col="grey", xlab="�A�C�e���̍w���p�x", main="�A�C�e���̍w���p�x���z")
gc(); gc()


##�f���x�N�g���𐶐�
k1 <- 3; k2 <- 5; k3 <- 5
x1 <- matrix(runif(hhpt*k1, 0, 1), nrow=hhpt, ncol=k1)
x2 <- matrix(0, nrow=hhpt, ncol=k2)
for(j in 1:k2){
  pr <- runif(1, 0.25, 0.55)
  x2[, j] <- rbinom(hhpt, 1, pr)
}
x3 <- rmnom(hhpt, 1, runif(k3, 0.2, 1.25)); x3 <- x3[, -which.min(colSums(x3))]
x <- cbind(1, x1, x2, x3)   #�f�[�^������
col_x <- ncol(x)

##�K�w���f���̐����ϐ��𐶐�
#���[�U�[�̐����ϐ�
k1 <- 2; k2 <- 4; k3 <- 5
u1 <- matrix(runif(hh*k1, 0, 1), nrow=hh, ncol=k1)
u2 <- matrix(0, nrow=hh, ncol=k2)
for(j in 1:k2){
  pr <- runif(1, 0.25, 0.55)
  u2[, j] <- rbinom(hh, 1, pr)
}
u3 <- rmnom(hh, 1, runif(k3, 0.2, 1.25)); u3 <- u3[, -which.min(colSums(u3))]
u <- cbind(1, u1, u2, u3)   #�f�[�^������
col_u <- ncol(u)

#�A�C�e���̐����ϐ�
k1 <- 2; k2 <- 4; k3 <- 4
g1 <- matrix(runif(item*k1, 0, 1), nrow=item, ncol=k1)
g2 <- matrix(0, nrow=item, ncol=k2)
for(j in 1:k2){
  pr <- runif(1, 0.25, 0.55)
  g2[, j] <- rbinom(item, 1, pr)
}
g3 <- rmnom(item, 1, runif(k3, 0.2, 1.25)); g3 <- g3[, -which.min(colSums(g3))]
g <- cbind(1, g1, g2, g3)   #�f�[�^������
col_g <- ncol(g)


####�g�s�b�N���f���̃f�[�^�𐶐�####
##�p�����[�^��ݒ�
#�f�B���N�����O���z�̃p�����[�^
alpha1 <- rep(0.15, k)
alpha2 <- rep(0.05, v)

#�S��ނ̒P�ꂪ�o������܂ŌJ��Ԃ�
rp <- 0
repeat {
  rp <- rp + 1
  print(rp)
  
  #�f�B���N�����z����p�����[�^�𐶐�
  theta <- thetat <- extraDistr::rdirichlet(item, alpha1)
  phi <- extraDistr::rdirichlet(k, alpha2)
  
  #�P��o���m�����Ⴂ�g�s�b�N�����ւ���
  index <- which(colMaxs(phi) < (k*10)/f)
  for(j in 1:length(index)){
    phi[as.numeric(rmnom(1, 1, extraDistr::rdirichlet(1, rep(2.0, k))) %*% 1:k), index[j]] <- (k*10)/f
  }
  phit <- phi

  ##�g�s�b�N�ƒP��𐶐�
  word_list <- Z_list <- list()
  WX <- matrix(0, nrow=item, ncol=v)
  
  for(i in 1:item){
    #�g�s�b�N�𐶐�
    z <- rmnom(w[i], 1, theta[i, ])
    z_vec <- as.numeric(z %*% 1:k)
    
    #�P��𐶐�
    word <- rmnom(w[i], 1, phi[z_vec, ])
    WX[i, ] <- colSums(word)
    
    #�f�[�^���i�[
    word_list[[i]] <- as.numeric(word %*% 1:v)
    Z_list[[i]] <- z
  }
  if(min(colSums(WX)) > 0) break
}

#���X�g��ϊ�
wd <- unlist(word_list)
Z <- do.call(rbind, Z_list)

#�X�p�[�X�s����쐬
d_data <- sparseMatrix(1:f, d_id, x=rep(1, f), dims=c(f, item))
d_data_T <- t(d_data)
word_data <- sparseMatrix(1:f, wd, x=rep(1, f), dims=c(f, v))
word_data_T <- t(word_data)

#�C���f�b�N�X��ݒ�
doc_list <- wd_list <- list()
for(i in 1:item){
  doc_list[[i]] <- which(d_id==i)
}
for(j in 1:v){
  wd_list[[j]] <- which(wd==j)
}

#�g�s�b�N���z�̊��Ғl��ݒ�
Z_score <- matrix(0, nrow=item, ncol=k) 
for(i in 1:item){
  Z_score[i, ] <- colMeans(Z[doc_list[[i]], ])
}


####�]���x�N�g���𐶐�####
rp <- 0
repeat { 
  rp <- rp + 1
  
  ##�p�����[�^��ݒ�
  #�f���x�N�g���̃p�����[�^
  sigma <- sigmat <- 0.5
  beta <- betat <- c(5.5, rnorm(col_x-1, 0, 0.75))
  
  #�K�w���f���̕��U�p�����[�^
  Cov_u <- Cov_ut <- runif(1, 0.1, 0.4)   #���[�U�[-�A�C�e���̊K�w���f���̕W���΍�
  Cov_v <- Cov_vt <- runif(1, 0.1, 0.4)   #�A�C�e���̊K�w���f���̕W���΍�
  Cov_z <- Cov_zt <- diag(runif(k, 0.01, 0.1), k)   #�g�s�b�N�̃��[�U�[�����x�N�g���̊K�w���f���̕��U
  
  #�K�w���f���̉�A�W����ݒ�
  alpha_u <- alpha_ut <- rnorm(col_u, 0, 0.35)
  alpha_v <- alpha_vt <- rnorm(col_g, 0, 0.35)
  alpha_z <- alpha_zt <- mvrnorm(col_u, rep(0, k), runif(k, 0.1, 0.25) * diag(k))
  
  #�ϗʌ��ʂƓ����x�N�g���̃p�����[�^�𐶐�
  theta_u <- theta_ut <- u %*% alpha_u + rnorm(hh, 0, Cov_u)
  theta_v <- theta_vt <- g %*% alpha_v + rnorm(item, 0, Cov_v)
  theta_z <- theta_zt <- u %*% alpha_z + mvrnorm(hh, rep(0, k), Cov_z)
  
  #���K���z����X�R�A�𐶐�
  mu <- as.numeric(x %*% beta + theta_u[user_id] + theta_v[item_id] + (Z_score[item_id, ] * theta_z[user_id, ]) %*% vec_k)
  y0 <- rnorm(hhpt, mu, sigma)
  
  #break����
  print(round(c(max(y0), min(y0)), 3))
  if(max(y0) < 15.0 & min(y0) > -4.0 & max(y0) > 11.0 & min(y0) < -1.0){
    break
  }
}

#���������X�R�A��]���f�[�^�ɕϊ�
y0_censor <- ifelse(y0 < 1, 1, ifelse(y0 > 10, 10, y0)) 
y <- round(y0_censor, 0)   #�X�R�A���ۂ߂�

#�X�R�A���z�ƒP�ꕪ�z
hist(y0, col="grey", breaks=25, xlab="�X�R�A", main="���S�f�[�^�̃X�R�A���z")
hist(y, col="grey", breaks=25, xlab="�X�R�A", main="�ϑ����ꂽ�X�R�A���z")


####�}���R�t�A�������e�J�����@��fLDA�𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
burden_fr <- function(theta, phi, wd, w, k, vec_k){
  #���S�W�����v�Z
  Bur <- theta[w, ] * t(phi)[wd, ]   #�ޓx
  Br <- Bur / as.numeric(Bur %*% vec_k)   #���S��
  bval <- list(Br=Br, Bur=Bur)
  return(bval)
}

##�A���S���Y���̐ݒ�
R <- 3000   #�T���v�����O��
keep <- 2   #2���1��̊����ŃT���v�����O���ʂ��i�[
disp <- 10
iter <- 0
burnin <- 1000/keep

##�C���f�b�N�X�ƃf�[�^�̐ݒ�
#�C���f�b�N�X��ݒ�
index_u <- index_w <- y_vec <- z_dt <- list()
for(j in 1:item){
  index_y <- item_list[[j]]; n <- item_n[j]
  index_u[[j]] <- rep(user_id[index_y], w[j]); index_w[[j]] <- rep(1:w[j], length(index_y))   #�C���f�b�N�X
  y_vec[[j]] <- rep(y[index_y], w[j])   #�X�R�A�̃x�N�g��
  z_dt[[j]] <- sparseMatrix(rep(1:w[j], rep(n, w[j])), 1:(n*w[j]), x=rep(1, n*w[j]), dims=c(w[j], n*w[j]))   #�a����邽�߂̑a�s��
}

#�f�[�^�̐ݒ�
xx <- t(x) %*% x
uu <- t(u) %*% u
gg <- t(g) %*% g


##���O���z�̐ݒ�
#�g�s�b�N���f���̎��O���z
alpha01 <- 0.1
alpha02 <- 0.1

#�s�񕪉��̎��O���z
beta01 <- 0
beta02 <- 0
s0 <- 0.1
v0 <- 0.1
Cov_x <- 100 * diag(col_x); inv_Cov_x <- solve(Cov_x)
tau_u <- 100 * diag(ncol(u)); inv_tau_u <- solve(tau_u)
tau_v <- 100 * diag(ncol(g)); inv_tau_g <- solve(tau_v)
Deltabar <- matrix(0, nrow=ncol(u), ncol=k)   #�K�w���f���̉�A�W���̎��O���z�̕���
ADelta <- 0.01 * diag(1, ncol(u))   #�K�w���f���̉�A�W���̎��O���z�̕��U
nu <- k + 1   #�t�E�B�V���[�g���z�̎��R�x
V <- nu * diag(rep(1, k)) #�t�E�B�V���[�g���z�̃p�����[�^


##�p�����[�^�̐^�l
#�g�s�b�N���f���̃p�����[�^
theta <- thetat
phi <- phit
wsum <- as.matrix(d_data_T %*% Z)
Z_score <- wsum / w

#�f���x�N�g���̃p�����[�^
sigma <- sigmat
beta <- betat
beta_mu <- as.numeric(x %*% beta)

#�K�w���f���̕��U�p�����[�^
Cov_u <- Cov_ut; inv_Cov_u <- 1 / Cov_u 
Cov_v <- Cov_vt; inv_Cov_v <- 1 / Cov_v 
Cov_z <- Cov_zt; inv_Cov_z <- solve(Cov_z)

#�K�w���f���̉�A�W����ݒ�
alpha_u <- alpha_ut; u_mu <- as.numeric(u %*% alpha_u)
alpha_v <- alpha_vt; v_mu <- as.numeric(g %*% alpha_v)
alpha_z <- alpha_zt; z_mu <- u %*% alpha_z

#�ϗʌ��ʂƓ����x�N�g���̃p�����[�^
theta_u <- theta_ut; theta_user <- theta_u[user_id]
theta_v <- theta_vt; theta_item <- theta_v[item_id]
theta_z <- theta_zt; theta_topic <- theta_z[user_id, ]


##�����l�̐ݒ�
#�g�s�b�N���f���̏����l
theta <- extraDistr::rdirichlet(item, rep(1.0, k))
phi <- extraDistr::rdirichlet(k, rep(1.0, v))
Zi <- rmnom(f, 1, rep(1/k, k))
wsum <- as.matrix(d_data_T %*% Z)
Z_score <- wsum / w

#�f���x�N�g���̃p�����[�^
sigma <- 1.0
beta <- as.numeric(solve(t(x) %*% x) %*% t(x) %*% y)
beta_mu <- as.numeric(x %*% beta)

#�K�w���f���̕��U�p�����[�^
Cov_u <- 0.2; inv_Cov_u <- 1 / Cov_u 
Cov_v <- 0.2; inv_Cov_v <- 1 / Cov_v
Cov_z <- 0.01 * diag(k); inv_Cov_z <- solve(Cov_z)

#�K�w���f���̉�A�W����ݒ�
alpha_u <- rnorm(col_u, 0, 0.1); u_mu <- as.numeric(u %*% alpha_u)
alpha_v <- rnorm(col_g, 0, 0.1); v_mu <- as.numeric(g %*% alpha_v)
alpha_z <- mvrnorm(col_u, rep(0, k), 0.01 * diag(k)); z_mu <- u %*% alpha_z

#�ϗʌ��ʂƓ����x�N�g���̃p�����[�^�𐶐�
theta_u <- u %*% alpha_u + rnorm(hh, 0, Cov_u); theta_user <- theta_u[user_id]
theta_v <- g %*% alpha_v + rnorm(item, 0, Cov_v); theta_item <- theta_v[item_id]
theta_z <- u %*% alpha_z + mvrnorm(hh, rep(0, k), Cov_z); theta_topic <- theta_z[user_id, ]


##�p�����[�^�̊i�[�p�z��
#�g�s�b�N���f���̃p�����[�^�̊i�[�p�z��
THETA <- array(0, dim=c(item, k, R/keep))
PHI <- array(0, dim=c(k, v, R/keep))
SEG <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG) <- "integer"

#���f���p�����[�^�̊i�[�p�z��
d <- 0
BETA <- matrix(0, nrow=R/keep, ncol=ncol(x))
SIGMA <- rep(0, R/keep)
THETA_U <- matrix(0, nrow=R/keep, ncol=hh)
THETA_V <- matrix(0, nrow=R/keep, ncol=item)
THETA_Z <- array(0, dim=c(hh, k, R/keep))

#�K�w���f���̊i�[�p�z��
ALPHA_U <- matrix(0, nrow=R/keep, ncol=col_u)
ALPHA_V <- matrix(0, nrow=R/keep, ncol=col_g)
ALPHA_Z <- array(0, dim=c(col_u, k, R/keep))
COV_U <- COV_V <- rep(0, R/keep)
COV_Z <- array(0, dim=c(k, k, R/keep))

##�ΐ��ޓx�̊�l
#1�p�����[�^���f��
LLst <- sum(dnorm(y, mean(y), sd(y), log=TRUE))

#�x�X�g���f���̑ΐ��ޓx
score <- as.matrix(d_data_T %*% Z / w)
uz <- rowSums(score[item_id, ] * theta_zt[user_id, ])
mu <- as.numeric(x %*% betat + theta_ut[user_id] + theta_vt[item_id] + uz)
LLbest <- sum(dnorm(y, mu, sigmat, log=TRUE))


####�M�u�X�T���v�����O���Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�g�s�b�N���Ƃ̕]���X�R�A�̖ޓx��ݒ�
  #�g�s�b�N���q���������X�R�A�̊��Ғl
  mu_uv <- as.numeric(x %*% beta + theta_user + theta_item)   #�g�s�b�N���q�����������Ғl
  wsum_z <- (wsum - rmnom(item, 1, wsum / as.numeric(wsum %*% vec_k)))   #�g�s�b�N���z�ɔ�Ⴕ�ăg�s�b�N�����O
  
  #�]���X�R�A�̃g�s�b�N�������Ƃ̑ΐ��ޓx
  LLi <- matrix(0, nrow=item, ncol=k)   #�ΐ��ޓx�̊i�[�p�z��
  for(j in 1:k){
    #�g�s�b�N�X�R�A�̊���
    wsum_z0 <- wsum_z; wsum_z0[, j] <- wsum_z0[, j] + 1
    z_score <- (wsum_z0 / w)[item_id, ]
    
    #�ΐ��ޓx���v�Z
    uz <- as.numeric((z_score * theta_topic) %*% vec_k)
    LLi[, j] <- as.numeric(item_data_T %*% dnorm(y, mu_uv + uz, sigma, log=TRUE))
  }
  
  ##�P��g�s�b�N���T���v�����O
  #�P�ꂲ�ƂɃg�s�b�N�̊����m����ݒ�
  Lho <- theta[d_id, ] * t(phi)[wd, ] * exp(LLi - rowMaxs(LLi))[d_id, ]   #�g�s�b�N�������Ƃ̖ޓx
  topic_rate <- Lho / as.numeric(Lho %*% vec_k)   #�g�s�b�N�̊����m��
  
  #�������z����g�s�b�N���T���v�����O
  Zi <- rmnom(f, 1, topic_rate)
  z_vec <- as.numeric(Zi %*% 1:k)
  
  
  ##�g�s�b�N���f���̃p�����[�^�𐄒�
  #�g�s�b�N���z���T���v�����O
  wsum <- as.matrix(d_data_T %*% Zi)
  theta <- extraDistr::rdirichlet(item, wsum + alpha01)
  Z_score <- wsum / w; z_score <- Z_score[item_id, ]
  
  #�P�ꕪ�z�̃T���v�����O
  vsum <- as.matrix(t(word_data_T %*% Zi)) + alpha02
  phi <- extraDistr::rdirichlet(k, vsum)
  
  
  ##�f���x�N�g���̃p�����[�^���T���v�����O
  #�����ϐ��̐ݒ�
  uz <- as.numeric((z_score * theta_topic) %*% vec_k)
  y_er <- y - theta_u[user_id] - theta_v[item_id] - uz
  
  #�f���x�N�g���̎��㕪�z�̃p�����[�^
  Xy <- t(x) %*% y_er
  inv_XXV <- solve(xx + inv_Cov_x)
  mu_vec <- inv_XXV %*% Xy   #���㕪�z�̕���
  
  #���ϗʐ��K���z����f���x�N�g�����T���v�����O
  beta <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  beta_mu <- as.numeric(x %*% beta)
  
  ##���f���̕W���΍����T���v�����O
  #�t�K���}���z�̃p�����[�^
  er <- y - beta_mu - theta_user - theta_item - uz   #���f���̌덷
  s1 <- as.numeric(t(er) %*% er) + s0
  v1 <- hhpt + v0
  
  #�t�K���}���z����W���΍����T���v�����O
  sigma <- sqrt(1/rgamma(1, v1/2, s1/2))
  
  
  ##���[�U�̕ϗʌ��ʂ��T���v�����O
  #���[�U�[�ϗʌ��ʂ̉����ϐ��̐ݒ�
  y_er <- y - beta_mu - theta_item - uz
  
  for(i in 1:hh){
    #���[�U�[�ϗʌ��ʂ̎��㕪�z�̃p�����[�^
    w_omega <- 1/Cov_u^2 + pt[i]/sigma^2
    weight <- (1/Cov_u^2) / w_omega
    mu_scalar <- weight*u_mu[i] + (1-weight)*mean(y_er[user_list[[i]]])
  
    #���K���z����p�����[�^�T���v�����O
    theta_u[i] <- rnorm(1, mu_scalar, 1/ w_omega)
  }
  theta_user <- theta_u[user_id]
  
  ##�A�C�e���̕ϗʌ��ʂ��T���v�����O
  #�A�C�e���ϗʌ��ʂ̉����ϐ��̐ݒ�
  y_er <- y - beta_mu - theta_user - uz
  
  for(j in 1:item){
    #�A�C�e���ϗʌ��ʂ̎��㕪�z�̃p�����[�^
    w_omega <- 1/Cov_v^2 + item_n[j]/sigma^2
    weight <- (1/Cov_v^2) / w_omega
    mu_scalar <- weight*v_mu[j] + (1-weight)*mean(y_er[item_list[[j]]])
    
    #���K���z���烆�[�U�[�ϗʌ��ʂ��T���v�����O
    theta_v[j] <- rnorm(1, mu_scalar, 1/ w_omega)
  }
  theta_item <- theta_v[item_id]
  
  
  ##���[�U�[���Ƃɓ����x�N�g�����T���v�����O
  #�����ϐ��̐ݒ�
  y_er <- y - beta_mu - theta_user - theta_item
  
  for(i in 1:hh){
    #�����x�N�g���̎��㕪�z�̃p�����[�^
    index <- user_list[[i]]   #���[�U�[�C���f�b�N�X
    X <- z_score[index, ]   #�g�s�b�N�̌o�����z
    Xy <- t(X) %*% y_er[index]
    inv_XXV <- solve(t(X) %*% X + inv_Cov_z)
    mu_vec <- inv_XXV %*% (Xy + inv_Cov_z %*% z_mu[i, ])   #���㕪�z�̕���
    
    #���ϗʐ��K���z��������x�N�g�����T���v�����O
    theta_z[i, ] <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  }
  theta_topic <- theta_z[user_id, ]
  uz <- as.numeric((z_score * theta_topic) %*% vec_k)
  
  
  ##���[�U�[�̕ϗʌ��ʂ̊K�w���f���̃p�����[�^���T���v�����O
  #���㕪�z�̃p�����[�^
  Xy <- t(u) %*% theta_u
  inv_XXV <- solve(t(u) %*% u + inv_tau_u)
  mu_vec <- inv_XXV %*% Xy   #���㕪�z�̕���
  
  #���ϗʐ��K���z����f���x�N�g�����T���v�����O
  alpha_u <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  u_mu <- as.numeric(u %*% alpha_u)
  
  #���f���̕W���΍����T���v�����O
  #�t�K���}���z�̃p�����[�^
  er <- theta_u - u_mu   #���f���̌덷
  s1 <- as.numeric(t(er) %*% er) + s0
  v1 <- hh + v0
  
  #�t�K���}���z����W���΍����T���v�����O
  Cov_u <- sqrt(1/rgamma(1, v1/2, s1/2))
  
  
  ##�A�C�e���̕ϗʌ��ʂ̊K�w���f���̃p�����[�^���T���v�����O
  #���㕪�z�̃p�����[�^
  Xy <- t(g) %*% theta_v
  inv_XXV <- solve(t(g) %*% g + inv_tau_g)
  mu_vec <- inv_XXV %*% Xy   #���㕪�z�̕���
  
  #���ϗʐ��K���z����f���x�N�g�����T���v�����O
  alpha_v <- mvrnorm(1, mu_vec, sigma^2*inv_XXV)
  v_mu <- as.numeric(g %*% alpha_v)
  
  #���f���̕W���΍����T���v�����O
  #�t�K���}���z�̃p�����[�^
  er <- theta_v - v_mu   #���f���̌덷
  s1 <- as.numeric(t(er) %*% er) + s0
  v1 <- item + v0
  
  #�t�K���}���z����W���΍����T���v�����O
  Cov_v <- sqrt(1/rgamma(1, v1/2, s1/2))
  
  
  ##���[�U�[�����s��̊K�w���f���̃p�����[�^�𐄒�
  #���ϗʉ�A���f������p�����[�^���T���v�����O
  out <- rmultireg(theta_z, u, Deltabar, ADelta, nu, V)
  alpha_z <- out$B; z_mu <- u %*% alpha_z   
  Cov_z <- out$Sigma; inv_Cov_z <- solve(Cov_z)
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̊i�[
  if(rp%%keep==0){
    mkeep <- rp/keep
    #�g�s�b�N���f���̃p�����[�^���i�[
    THETA[, , mkeep] <- theta
    PHI[, , mkeep] <- phi
    
    #���f���p�����[�^�̊i�[
    BETA[mkeep, ] <- beta
    SIGMA[mkeep] <- sigma
    THETA_U[mkeep, ] <- theta_u
    THETA_V[mkeep, ] <- theta_v
    THETA_Z[, , mkeep] <- theta_z
    
    #�K�w���f���̊i�[
    ALPHA_U[mkeep, ] <- alpha_u
    ALPHA_V[mkeep, ] <- alpha_v
    ALPHA_Z[, , mkeep] <- alpha_z 
    COV_U[mkeep] <- Cov_u
    COV_V[mkeep] <- Cov_v
    COV_Z[, , mkeep] <- Cov_z

    if(rp >= burnin){
      d <- d + 1
      SEG <- SEG + Zi
    }
  }
  
  if(rp%%disp==0){
    #�ΐ��ޓx���v�Z
    mu <- beta_mu + theta_user + theta_item + uz   #���Ғl
    LL <- sum(dnorm(y, mu, sigma, log=TRUE))   #�ΐ��ޓx�̘a 
    
    #�T���v�����O���ʂ̕\��
    print(rp)
    print(c(LL, LLbest, LLst))
    print(round(rbind(beta, betat), 3))
  }
}
