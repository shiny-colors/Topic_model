#####Symmentric Correspondence LDA#####
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
####�f�[�^�̔���####
##�f�[�^�̐ݒ�
L <- 3   #�f�[�^��
k <- 15   #�g�s�b�N��
d <- 2000   #������
v1 <- 1000; v2 <- 800; v3 <- 700   #��b��
max_v <- max(c(v1, v2, v3))
w1 <- rpois(d, rgamma(d, 85, 0.6)); w2 <- rpois(d, rgamma(d, 80, 0.65)); w3 <- rpois(d, rgamma(d, 75, 0.65))   #�P�ꐔ
w <- list(w1, w2, w3)
f1 <- sum(w1); f2 <- sum(w2); f3 <- sum(w3)   #���P�ꐔ
f <- c(f1, f2, f3)

#ID�̐ݒ�
d_id1 <- rep(1:d, w1)
d_id2 <- rep(1:d, w2)
d_id3 <- rep(1:d, w3)

##�f�[�^�̐���
for(rp in 1:1000){
  print(rp)
  
  #�f�B�N�������z�̃p�����[�^
  alpha1 <- rep(0.4, L)
  alpha2 <- rep(0.15, k)
  beta1 <- rep(0.1, v1); beta2 <- rep(0.1, v2); beta3 <- rep(0.1, v3)
  
  #�f�B�N�������z����p�����[�^�𐶐�
  theta1 <- thetat1 <- extraDistr::rdirichlet(d, alpha1)
  theta2 <- thetat2 <- array(0, dim=c(d, k, L))
  for(j in 1:L){
    theta2[, , j] <- thetat2[, , j] <- extraDistr::rdirichlet(d, alpha2)
  }
  phi1 <- phit1 <- extraDistr::rdirichlet(k, beta1)
  phi2 <- phit2 <- extraDistr::rdirichlet(k, beta2)
  phi3 <- phit3 <- extraDistr::rdirichlet(k, beta3)
  phi <- list(phi1, phi2, phi3)
  
  ##�����ߒ��Ɋ�Â��P��𐶐�
  WX_list <- list()
  word_list <- list()
  Z1_list <- list()
  Z2_list <- list()
  
  for(l in 1:L){
    print(l)
    Z1 <- Z2 <- WX <- word_data <- list()
    
    for(i in 1:d){
      n <- w[[l]][i]
      
      #���[�h�ϐ��𐶐�
      z1 <- rmnom(n, 1, theta1[i, ])
      z1_vec <- as.numeric(z1 %*% 1:L)
      
      #�g�s�b�N�𐶐�
      z2 <- rmnom(n, 1, t(theta2[i, , z1_vec]))
      z2_vec <- as.numeric(z2 %*% 1:k)
      
      #�P��𐶐�
      words <- rmnom(n, 1, phi[[l]][z2_vec, ])
      
      #�������Ƃ̃f�[�^���i�[
      Z1[[i]] <- z1
      Z2[[i]] <- z2
      WX[[i]] <- colSums(words)
      word_data[[i]] <- words
    }
    #�S�̂̃f�[�^���i�[
    Z1_list[[l]] <- do.call(rbind, Z1)
    Z2_list[[l]] <- do.call(rbind, Z2)
    WX_list[[l]] <- do.call(rbind, WX)
    word_list[[l]] <- do.call(rbind, word_data)
    rm(word_data); rm(WX); rm(Z1); rm(Z2)
  }
  #�P��o�����̍ŏ��l��0�łȂ��Ȃ�break
  if(min(unlist(lapply(WX_list, colSums))) > 0){
    break
  }
}

#���X�g��ϊ�
Z11 <- Z1_list[[1]]; Z12 <- Z1_list[[2]]; Z13 <- Z1_list[[3]]
Z21 <- Z2_list[[1]]; Z22 <- Z2_list[[2]]; Z23 <- Z2_list[[3]]
WX1 <- WX_list[[1]]; WX2 <- WX_list[[2]]; WX3 <- WX_list[[3]]
word_data1 <- as(word_list[[1]], "CsparseMatrix")
word_data2 <- as(word_list[[2]], "CsparseMatrix")
word_data3 <- as(word_list[[3]], "CsparseMatrix")
storage.mode(WX1) <- "integer"
storage.mode(WX2) <- "integer"
storage.mode(WX3) <- "integer"
wd1 <- as.numeric(word_data1 %*% 1:v1)
wd2 <- as.numeric(word_data2 %*% 1:v2)
wd3 <- as.numeric(word_data3 %*% 1:v3)
rm(Z1_list); rm(Z2_list); rm(word_list); rm(WX_list)
gc(); gc()


####�}���R�t�A�������e�J�����@��SymCorrLDA�𐄒�####
burden_fr <- function(theta, phi, wd, w, k){
  #���S�W�����v�Z
  Bur <- theta[w, ] * t(phi)[wd, ]   #�ޓx
  Br <- Bur / rowSums(Bur)   #���S��
  r <- colSums(Br) / sum(Br)   #������
  bval <- list(Br=Br, Bur=Bur, r=r)
  return(bval)
}

##�A���S���Y���̐ݒ�
R <- 5000
keep <- 2  
iter <- 0
burnin <- 1000/keep
disp <- 10

##�C���f�b�N�X�̐ݒ�
doc1_list <- doc2_list <- doc3_list <- list()
doc1_vec <- doc2_vec <- doc3_vec <- list()
wd1_list <- wd2_list <- wd3_list <- list()
wd1_vec <- wd2_vec <- wd3_vec <- list()

for(i in 1:d){
  doc1_list[[i]] <- which(d_id1==i)
  doc2_list[[i]] <- which(d_id2==i)
  doc3_list[[i]] <- which(d_id3==i)
  doc1_vec[[i]] <- rep(1, length(doc1_list[[i]]))
  doc2_vec[[i]] <- rep(1, length(doc2_list[[i]]))
  doc3_vec[[i]] <- rep(1, length(doc3_list[[i]]))
}
for(j in 1:v1){
  wd1_list[[j]] <- which(wd1==j)
  wd1_vec[[j]] <- rep(1, length(wd1_list[[j]]))
}
for(j in 1:v2){
  wd2_list[[j]] <- which(wd2==j)
  wd2_vec[[j]] <- rep(1, length(wd2_list[[j]]))
}
for(j in 1:v3){
  wd3_list[[j]] <- which(wd3==j)
  wd3_vec[[j]] <- rep(1, length(wd3_list[[j]]))
}

##���O���z�̐ݒ�
alpha1 <- 10
alpha2 <- 1
beta1 <- 0.1   

##�p�����[�^�̐^�l
theta1 <- thetat1
theta2 <- thetat2
phi1 <- phit1 
phi2 <- phit2
phi3 <- phit3

##�p�����[�^�̏����l
theta1 <- extraDistr::rdirichlet(d, rep(1.0, L))
theta2 <- array(0, dim=c(d, k, L))
for(j in 1:L){
  theta2[, , j] <- extraDistr::rdirichlet(d, rep(1.0, k))
}
phi1 <- extraDistr::rdirichlet(k, rep(1.0, v1))
phi2 <- extraDistr::rdirichlet(k, rep(1.0, v2))
phi3 <- extraDistr::rdirichlet(k, rep(1.0, v3))


##�p�����[�^�̊i�[�p�z��
THETA1 <- array(0, dim=c(d, L, R/keep))
THETA21 <- array(0, dim=c(d, k, R/keep))
THETA22 <- array(0, dim=c(d, k, R/keep))
THETA23 <- array(0, dim=c(d, k, R/keep))
PHI1 <- array(0, dim=c(k, v1, R/keep))
PHI2 <- array(0, dim=c(k, v2, R/keep))
PHI3 <- array(0, dim=c(k, v3, R/keep))
SEG11 <- matrix(0, nrow=f1, ncol=L)
SEG12 <- matrix(0, nrow=f2, ncol=L)
SEG13 <- matrix(0, nrow=f3, ncol=L)
SEG21 <- matrix(0, nrow=f1, ncol=k)
SEG22 <- matrix(0, nrow=f2, ncol=k)
SEG23 <- matrix(0, nrow=f3, ncol=k)


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##���[�h�ϐ����T���v�����O
  #���[�h�ϐ��̖ޓx���v�Z
  LLi1 <- matrix(0, nrow=f1, ncol=L); LLi2 <- matrix(0, nrow=f2, ncol=L); LLi3 <- matrix(0, nrow=f3, ncol=L)
  phi_par1 <- t(phi1)[wd1, ]; phi_par2 <- t(phi2)[wd2, ]; phi_par3 <- t(phi3)[wd3, ]   #�o���P��̊m��
  
  #���[�h���Ƃ̖ޓx�̊��Ғl
  theta_par1 <- theta_par2 <- theta_par3 <- list()
  
  for(j in 1:L){
    theta_par1[[j]] <- theta2[d_id1, , j]
    theta_par2[[j]] <- theta2[d_id2, , j]
    theta_par3[[j]] <- theta2[d_id3, , j]
    LLi1[, j] <- rowSums(theta_par1[[j]] * phi_par1)
    LLi2[, j] <- rowSums(theta_par2[[j]] * phi_par2)
    LLi3[, j] <- rowSums(theta_par3[[j]] * phi_par3)
  }
  
  #���ݕϐ�z�̌v�Z
  mode_par1 <- theta1[d_id1, ] * LLi1; mode_par2 <- theta1[d_id2, ] * LLi2; mode_par3 <- theta1[d_id3, ] * LLi3
  z11_rate <- mode_par1 / rowSums(mode_par1)
  z12_rate <- mode_par2 / rowSums(mode_par2)
  z13_rate <- mode_par3 / rowSums(mode_par3)
  
  #�������z��胂�[�h�ϐ����T���v�����O
  Zi11 <- rmnom(f1, 1, z11_rate)
  Zi12 <- rmnom(f2, 1, z12_rate)
  Zi13 <- rmnom(f3, 1, z13_rate)
  Zi11_T <- t(Zi11); Zi12_T <- t(Zi12); Zi13_T <- t(Zi13)
  
  
  ##�g�s�b�N���T���v�����O
  #���[�h�ϐ��̊����Ɋ�Â��g�s�b�N���z���v�Z
  theta_topic1 <- theta_par1[[1]] * Zi11[, 1] + theta_par1[[2]] * Zi11[, 2] + theta_par1[[3]] * Zi11[, 3]
  theta_topic2 <- theta_par2[[1]] * Zi12[, 1] + theta_par2[[2]] * Zi12[, 2] + theta_par2[[3]] * Zi12[, 3]
  theta_topic3 <- theta_par3[[1]] * Zi13[, 1] + theta_par3[[2]] * Zi13[, 2] + theta_par3[[3]] * Zi13[, 3]
  
  #���[�h���Ƃ̃g�s�b�N���z�̖ޓx
  topic_par1 <- theta_topic1 * phi_par1
  topic_par2 <- theta_topic2 * phi_par2
  topic_par3 <- theta_topic3 * phi_par3
  
  #���ݕϐ�z�̌v�Z
  z21_rate <- topic_par1 / rowSums(topic_par1)
  z22_rate <- topic_par2 / rowSums(topic_par2)
  z23_rate <- topic_par3 / rowSums(topic_par3)
  
  #�������z���g�s�b�N���T���v�����O
  Zi21 <- rmnom(f1, 1, z21_rate)
  Zi22 <- rmnom(f2, 1, z22_rate)
  Zi23 <- rmnom(f3, 1, z23_rate)
  Zi21_T <- t(Zi21); Zi22_T <- t(Zi22); Zi23_T <- t(Zi23)
  
  
  ##���[�h���z���p�����[�^���T���v�����O
  #�f�B�N�������z�̃p�����[�^���v�Z
  wsum01 <- matrix(0, nrow=d, ncol=L)
  for(i in 1:d){
    wsum01[i, ] <- Zi11_T[, doc1_list[[i]]] %*% doc1_vec[[i]] + Zi12_T[, doc2_list[[i]]] %*% doc2_vec[[i]] + 
      Zi13_T[, doc3_list[[i]]] %*% doc3_vec[[i]]
  }
  wsum1 <- wsum01 + alpha1
  
  #�f�B�N�������z���p�����[�^���T���v�����O
  theta1 <- extraDistr::rdirichlet(d, wsum1)

    
  ##�g�s�b�N���z�̃p�����[�^���T���v�����O
  #�f�B�N�������z�̃p�����[�^���v�Z
  zi21_t <- zi22_t <- zi23_t <- list()
  for(j in 1:L){
    zi21_t[[j]] <- Zi21_T * matrix(Zi11_T[j, ], nrow=k, ncol=f[1], byrow=T)
    zi22_t[[j]] <- Zi22_T * matrix(Zi12_T[j, ], nrow=k, ncol=f[2], byrow=T)
    zi23_t[[j]] <- Zi23_T * matrix(Zi13_T[j, ], nrow=k, ncol=f[3], byrow=T)
  }
  
  wsum021 <- wsum022 <- wsum023 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum021[i, ] <- zi21_t[[1]][, doc1_list[[i]]] %*% doc1_vec[[i]] + zi22_t[[1]][, doc2_list[[i]]] %*% doc2_vec[[i]] +
      zi23_t[[1]][, doc3_list[[i]]] %*% doc3_vec[[i]]
    wsum022[i, ] <- zi21_t[[2]][, doc1_list[[i]]] %*% doc1_vec[[i]] + zi22_t[[2]][, doc2_list[[i]]] %*% doc2_vec[[i]] +
      zi23_t[[2]][, doc3_list[[i]]] %*% doc3_vec[[i]]
    wsum023[i, ] <- zi21_t[[3]][, doc1_list[[i]]] %*% doc1_vec[[i]] + zi22_t[[3]][, doc2_list[[i]]] %*% doc2_vec[[i]] +
      zi23_t[[3]][, doc3_list[[i]]] %*% doc3_vec[[i]]
  }
  
  #�f�B�N�������z���p�����[�^���T���v�����O
  theta2[, , 1] <- extraDistr::rdirichlet(d, wsum021 + alpha2)
  theta2[, , 2] <- extraDistr::rdirichlet(d, wsum022 + alpha2)
  theta2[, , 3] <- extraDistr::rdirichlet(d, wsum023 + alpha2)
  
  ##�P�ꕪ�z�̃p�����[�^���T���v�����O
  #�f�B�N�������z�̃p�����[�^���v�Z
  vsum01 <- matrix(0, nrow=k, ncol=v1); vsum02 <- matrix(0, nrow=k, ncol=v2); vsum03 <- matrix(0, nrow=k, ncol=v3)
  for(j in 1:max_v){
    if(v1 >= j){
      vsum01[, j] <- Zi21_T[, wd1_list[[j]], drop=FALSE] %*% wd1_vec[[j]]
    }
    if(v2 >= j){
      vsum02[, j] <- Zi22_T[, wd2_list[[j]], drop=FALSE] %*% wd2_vec[[j]]
    }
    if(v3 >= j){
      vsum03[, j] <- Zi23_T[, wd3_list[[j]], drop=FALSE] %*% wd3_vec[[j]]
    }
  }
  
  #�f�B�N�������z����p�����[�^���T���v�����O
  phi1 <- extraDistr::rdirichlet(k, vsum01+beta1)
  phi2 <- extraDistr::rdirichlet(k, vsum02+beta1)
  phi3 <- extraDistr::rdirichlet(k, vsum03+beta1)
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    THETA1[, , mkeep] <- theta1
    THETA21[, , mkeep] <- theta2[, , 1]
    THETA22[, , mkeep] <- theta2[, , 2]
    THETA23[, , mkeep] <- theta2[, , 3]
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi2
    PHI3[, , mkeep] <- phi3
    
    #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
    if(rp%%keep==0 & rp >= burnin){
      SEG11 <- SEG11 + Zi11; SEG12 <- SEG12 + Zi12; SEG13 <- SEG13 + Zi13
      SEG21 <- SEG21 + Zi21; SEG22 <- SEG22 + Zi22; SEG23 <- SEG23 + Zi23
    }
    
    if(rp%%disp==0){
      #�T���v�����O���ʂ��m�F
      print(rp)
      LL1 <- sum(log(rowSums(topic_par1)))
      LL2 <- sum(log(rowSums(topic_par2)))
      LL3 <- sum(log(rowSums(topic_par3)))
      print(c(LL1+LL2+LL3, LL1, LL2, LL3))
      print(round(cbind(theta1[1:10, ], thetat1[1:10, ]), 3))
      print(round(cbind(phi1[, 1:10], phit1[, 1:10]), 3))
    }
  }
}