#####�f�B���N�g��LDA���f��#####
options(warn=0)
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
#set.seed(2506787)

####�f�[�^�̔���####
##�f�[�^�̐ݒ�
k <- 10   #�g�s�b�N��
dir1 <- 5   #��ʃf�B���N�g����
dir2 <- 20   #���ʃf�B���N�g����
d <- 15000   #������
v11 <- 150   #�f�B���N�g��������̌�b��
v12 <- dir1*v11   #�f�B���N�g���\���Ɋ֌W�̂����b��
v2 <- 500   #�f�B���N�g���\���Ɋ֌W�̂Ȃ���b��
v <- v12 + v2   #����b��
index_v11 <- matrix(1:v12, nrow=dir1, ncol=v11, byrow=T)
dir_v <- matrix(1:v12, nrow=dir1, ncol=v11, byrow=T)   #�f�B���N�g���̒P��\��
w <- rpois(d, rgamma(d, 65, 0.5))   #����������̒P�ꐔ
f <- sum(w)   #���P�ꐔ

##ID�̐ݒ�
d_id <- rep(1:d, w)
t_id_list <- list()
for(i in 1:d){
  t_id_list[[i]] <- 1:w[i]
}
t_id <- unlist(t_id_list)

##�f�B���N�g���̊�����ݒ�
#��ʂƉ��ʂ̃f�B���N�g��������ݒ�
dir_sets1 <- matrix(1:dir2, nrow=dir1, ncol=dir2/dir1, byrow=T)
dir_sets2 <- rep(1:dir1, rep(dir2/dir1, dir1))
dir_freq <- rtpois(d, 0.7, 0, 3)   #����������̃f�B���N�g����
dir_id <- rep(1:d, dir_freq)   #�f�B���N�g����id
dir_n <- length(dir_id)
dir_index <- list()
for(i in 1:d){
  dir_index[[i]] <- which(dir_id==i)
}

#�f�B���N�g���̐���
dir_data1 <- matrix(0, nrow=dir_n, ncol=dir1)
dir_data2 <- matrix(0, nrow=dir_n, ncol=dir2)
for(i in 1:d){
  repeat{
    x <- rmnom(dir_freq[i], 1, rep(1, dir2))
    if(max(colSums(x))==1){
      index <- which(dir_id==i)
      x <- x[order(as.numeric(x %*% 1:dir2)), , drop=FALSE]
      dir_data1[index, ] <- t(apply(x, 1, function(y) tapply(y, dir_sets2, sum)))
      dir_data2[index, ] <- x
      break
    }
  }
}
#�f�B���N�g�����x�N�g���ɕϊ�
dir_vec1 <- as.numeric(dir_data1 %*% 1:dir1)
dir_vec2 <- as.numeric(dir_data2 %*% 1:dir2)


##�p�����[�^�̐ݒ�
#�f�B���N�����z�̎��O���z
alpha11 <- rep(0.15, k)
alpha21 <- c(rep(0.025, length(1:v12)), rep(0.15, length(1:v2)))
alpha22 <- matrix(0, nrow=dir2, ncol=v)
for(j in 1:dir1){
  x <- rep(0.001, v)
  x[index_v11[j, ]] <- 0.2
  alpha22[dir_sets1[j, ], ] <- matrix(x, nrow=length(dir_sets1[j, ]), ncol=v, byrow=T)
}

##���ׂĂ̒P�ꂪ�o������܂Ńf�[�^�̐����𑱂���
for(rp in 1:1000){
  print(rp)
  
  #�f�B���N�����z����p�����[�^�𐶐�
  theta <- thetat <- extraDistr::rdirichlet(d, alpha11)
  phi1 <- phit1 <- extraDistr::rdirichlet(k, alpha21)
  phi2 <- phit2 <- extraDistr::rdirichlet(dir2, alpha22)
  
  #�X�C�b�`���O�ϐ��𐶐�
  gamma_list <- list()
  for(i in 1:d){
    if(dir_freq[i]==1){
      par <- c(6.25, 5.0)
      gamma_list[[i]] <- rbeta(1, par[1], par[2])
    } else {
      par <- c(5.0, runif(dir_freq[i], 1.0, 4.5))
      gamma_list[[i]] <- as.numeric(extraDistr::rdirichlet(1, par))
    }
  }
  
  ##���f���Ɋ�Â��f�[�^�𐶐�
  word_list <- wd_list <- Z1 <- z1_list <- z2_list <- list()
  WX <- matrix(0, nrow=d, ncol=v)
  
  for(i in 1:d){
    #�X�C�b�`���O�ϐ��𐶐�
    n <- dir_freq[i] + 1
    if(dir_freq[i]==1){
      z1 <- rbinom(w[i], 1, gamma_list[[i]])
      Z1[[i]] <- cbind(z1, 1-z1)
      z1_list[[i]] <- as.numeric((Z1[[i]] * matrix(c(1, dir_vec2[dir_index[[i]]]), nrow=w[i], ncol=n, byrow=T)) %*% rep(1, n))
    } else {
      Z1[[i]] <- rmnom(w[i], 1, gamma_list[[i]])
      z1_list[[i]] <- as.numeric((Z1[[i]] * matrix(c(1, dir_vec2[dir_index[[i]]]), nrow=w[i], ncol=n, byrow=T)) %*% rep(1, n))
    }
    
    #�������z���g�s�b�N�𐶐�
    z2 <- matrix(0, nrow=w[i], ncol=k)
    index <- which(Z1[[i]][, 1]==1)
    z2[index, ] <- rmnom(length(index), 1, theta[i, ])
    z2_vec <- as.numeric(z2 %*% 1:k)
    
    #�g�s�b�N����уf�B���N�g������P��𐶐�
    word <- matrix(0, nrow=w[i], ncol=v)
    word[index, ] <- rmnom(length(index), 1, phi1[z2_vec[index], ])   #�g�s�b�N����P��𐶐�
    word[-index, ] <- rmnom(w[i]-length(index), 1, phi2[z1_list[[i]][-index], ])   #�f�B���N�g������P��𐶐�
    wd <- as.numeric(word %*% 1:v)
    storage.mode(word) <- "integer"
    
    #�f�[�^���i�[
    z2_list[[i]] <- z2
    wd_list[[i]] <- wd
    word_list[[i]] <- word
    WX[i, ] <- colSums(word)
  }
  #�S�P�ꂪ�o�����Ă�����break
  if(min(colSums(WX) > 0)) break
}

##���X�g��ϊ�
wd <- unlist(wd_list)
Z2 <- do.call(rbind, z2_list)
z2_vec <- as.numeric(Z2 %*% 1:k)
sparse_data <- sparseMatrix(1:f, wd, dims=c(f, v))
sparse_data_T <- t(sparse_data)
rm(word_list); rm(wd_list); rm(z2_list)
gc(); gc()

#�X�C�b�`���O�ϐ��̐^�l
ZT1_list <- list()
for(i in 1:d){
  ZT1_list[[i]] <- Z1[[i]][, 1]
}
ZT1 <- unlist(ZT1_list)

#####�}���R�t�A�������e�J�����@��DLDA�𐄒�####
##�P�ꂲ�Ƃɖޓx�ƕ��S�����v�Z����֐�
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

##���O���z�̐ݒ�
alpha1 <- 0.1
alpha2 <- 0.1
beta1 <- 1
beta2 <- 1

##�^�l�̐ݒ�
theta <- thetat
phi1 <- phit1
phi2 <- phit2
gamma <- gamma_list

##�����l�̐ݒ�
#�g�s�b�N���z�̏����l
theta <- extraDistr::rdirichlet(d, rep(1.0, k))
phi1 <- extraDistr::rdirichlet(k, rep(1.0, v))
phi2 <- extraDistr::rdirichlet(dir2, rep(1.0, v))

#�X�C�b�`���O���z�̏����l
gamma <- list()
for(i in 1:d){
  if(dir_freq[i]==1){
    gamma[[i]] <- 0.5
  } else {
    n <- dir_freq[i]+1
    gamma[[i]] <- rep(1/n, n)
  }
}

##�p�����[�^�̕ۑ��p�z��
THETA <- array(0, dim=c(d, k, R/keep))
PHI1 <- array(0, dim=c(k, v, R/keep))
PHI2 <- array(0, dim=c(dir2, v, R/keep))
SEG11 <- rep(0, f)
SEG12 <- matrix(0, nrow=f, ncol=dir2)
SEG2 <- matrix(0, nrow=f, ncol=k)
storage.mode(SEG12) <- "integer"
storage.mode(SEG2) <- "integer"

##�C���f�b�N�X��ݒ�
doc_list <- doc_vec <- list()
wd_list <- wd_vec <- list()
for(i in 1:d){
  doc_list[[i]] <- which(d_id==i)
  doc_vec[[i]] <- rep(1, length(doc_list[[i]]))
}
for(j in 1:v){
  wd_list[[j]] <- which(wd==j)
  wd_vec[[j]] <- rep(1, length(wd_list[[j]]))
}

##�f�[�^�̐ݒ�
dir_z <- matrix(0, nrow=d, ncol=dir2)
dir_list2 <- dir_list1 <- list()
for(i in 1:d){
  dir_z[i, ] <- colSums(dir_data2[dir_index[[i]], , drop=FALSE])
  dir_list1[[i]] <- (dir_z[i, ] * 1:dir2)[dir_z[i, ] > 0]
  dir_list2[[i]] <- matrix(dir_list1[[i]], nrow=w[i], ncol=dir_freq[i], byrow=T)
}
dir_Z <- dir_z[d_id, ]
storage.mode(dir_Z) <- "integer"

##�ΐ��ޓx�̊�l
LLst <- sum(sparse_data %*% log(colMeans(sparse_data)))


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�P�ꂲ�Ƃ̃X�C�b�`���O�ϐ��𐶐�
  #�g�s�b�N�ƃf�B���N�g���̊��Җޓx
  Lho1 <- theta[d_id, ] * t(phi1)[wd, ]
  Li1 <- rowSums(Lho1)   #�g�s�b�N�̊��Җޓx
  Li2 <- t(phi2)[wd, ] * dir_Z   #�f�B���N�g���̖ޓx
  
  #�x���k�[�C���z���邢�͑������z���X�C�b�`���O�ϐ��𐶐�
  Zi11 <- Zi12 <- rep(0, f)
  Lho_list <- list()
  
  for(i in 1:d){
    if(dir_freq[i]==1){
  
      #���ݕϐ�z�̐ݒ�
      omega <- matrix(c(gamma[[i]], 1-gamma[[i]]), nrow=w[i], ncol=dir_freq[i]+1, byrow=T)
      z_par <- omega * cbind(Li1[doc_list[[i]]], Li2[doc_list[[i]], dir_list1[[i]]])
      z_rate <- z_par[, 1] / rowSums(z_par)
      Lho_list[[i]] <- z_par
      
      
      #�x���k�[�C���z���X�C�b�`���O�ϐ��𐶐�
      z1 <- rbinom(w[i], 1, z_rate)
      Zi11[doc_list[[i]]] <- z1   #�g�s�b�N�Ɋ֌W�̂���P��
      Zi12[doc_list[[i]]] <- (1-z1) * dir_list1[[i]]   #�f�B���N�g���Ɋ֌W�̂���P��
      
      #�x�[�^���z���獬�������T���v�����O
      z_freq <- t(z1) %*% doc_vec[[i]]
      gamma[[i]] <- rbeta(1, z_freq+beta1, w[i]-z_freq+beta2)
  
    } else {
      
      #���ݕϐ�z�̐ݒ�
      omega <- matrix(gamma[[i]], nrow=w[i], ncol=dir_freq[i]+1, byrow=T)
      z_par <- omega * cbind(Li1[doc_list[[i]]], Li2[doc_list[[i]], dir_list1[[i]]])
      z_rate <- z_par / rowSums(z_par)
      Lho_list[[i]] <- z_par
      
      z1 <- rmnom(w[i], 1, z_rate)   #�X�C�b�`���O�ϐ��𐶐�
      Zi11[doc_list[[i]]] <- z1[, 1]   #�g�s�b�N�Ɋ֌W�̂���P��
      Zi12[doc_list[[i]]] <- (z1[, -1] * dir_list2[[i]]) %*% rep(1, length(dir_list1[[i]]))   #�f�B���N�g���Ɋ֌W�̂���P��
      
      #�f�B���N�����z���獬�������T���v�����O�u
      z_freq <- as.numeric(t(z1) %*% doc_vec[[i]])
      gamma[[i]] <- as.numeric(extraDistr::rdirichlet(1, as.numeric(t(z1) %*% doc_vec[[i]]) + alpha1))
    }
  }
  #���������X�C�b�`���O�ϐ��̃C���f�b�N�X���쐬
  index_z11 <- which(Zi11==1)
  
  
  ##�g�s�b�N���z����g�s�b�N���T���v�����O
  #�������z���g�s�b�N���T���v�����O
  Zi2 <- matrix(0, nrow=f, ncol=k)
  z_rate <- Lho1[index_z11, ] / Li1[index_z11]   #�g�s�b�N�̊����m��
  Zi2[index_z11, ] <- rmnom(length(index_z11), 1, z_rate)   #�g�s�b�N���T���v�����O
  Zi2_T <- t(Zi2)
  
  
  ##�g�s�b�N���z�̃p�����[�^���T���v�����O
  wsum0 <- matrix(0, nrow=d, ncol=k)
  for(i in 1:d){
    wsum0[i, ] <- Zi2_T[, doc_list[[i]], drop=FALSE] %*% doc_vec[[i]]
  }
  wsum <- wsum0 + alpha1   #�f�B���N�����z�̃p�����[�^
  theta <- extraDistr::rdirichlet(d, wsum)   #�p�����[�^���T���v�����O
  
  
  ##�P�ꕪ�z�̃p�����[�^���T���v�����O
  #�g�s�b�N�̒P�ꕪ�z���T���v�����O
  vsum0 <- matrix(0, nrow=k, ncol=v)
  for(j in 1:v){
    vsum0[, j] <- Zi2_T[, wd_list[[j]], drop=FALSE] %*% wd_vec[[j]]
  }
  vsum <- vsum0 + alpha2   
  phi1 <- extraDistr::rdirichlet(k, vsum)   #�p�����[�^���T���v�����O
  
  #�f�B���N�g���̒P�ꕪ�z���T���v�����O
  Zi0 <- Zi12[-index_z11]
  sparse_data0 <- sparse_data[-index_z11, ]
  dsum0 <- matrix(0, nrow=dir2, ncol=v)
  for(j in 1:dir2){
    dsum0[j, ] <- colSums(sparse_data0[Zi0==j, ]) 
  }
  dsum <- dsum0 + alpha2   #�f�B���N�����z�̃p�����[�^
  phi2 <- extraDistr::rdirichlet(dir2, dsum)   #�p�����[�^���T���v�����O
  
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^���i�[
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    PHI1[, , mkeep] <- phi1
    PHI2[, , mkeep] <- phi2
    THETA[, , mkeep] <- theta
  }  
  
  #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
  if(rp%%keep==0 & rp >= burnin){
    SEG0 <- matrix(0, nrow=f, ncol=dir2)
    for(i in 1:f){
      if(Zi12[i]==0) next
      SEG0[i, Zi12[i]] <- 1
    }
    SEG11 <- SEG11 + Zi11
    SEG12 <- SEG12 + SEG0
    SEG2 <- SEG2 + Zi2
  }
  
  if(rp%%disp==0){
    #�ΐ��ޓx���v�Z
    Lho <- rep(0, d)
    for(i in 1:d){
      Lho[i] <- sum(log(rowSums(Lho_list[[i]])))
    }
    #�T���v�����O���ʂ��m�F
    print(rp)
    print(c(sum(Lho), LLst))
    print(mean(Zi11))
    print(round(cbind(phi2[, (v11-4):(v11+5)], phit2[, (v11-4):(v11+5)]), 3))
  }
}

####�T���v�����O���ʂ̉����Ɨv��####
burnin <- 2000/keep
RS <- R/keep

##�T���v�����O���ʂ̉���
#�g�s�b�N���z�̉���
matplot(t(THETA[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[10, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[100, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(THETA[1000, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

#�P�ꕪ�z�̉���
matplot(t(PHI1[1, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI1[3, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI1[5, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI1[7, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[2, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[4, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[6, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")
matplot(t(PHI2[8, , ]), type="l", xlab="�T���v�����O��", ylab="�p�����[�^")

##�T���v�����O���ʂ̎��㕪�z
#�g�s�b�N���z�̎��㕽��
round(cbind(apply(THETA[, , burnin:RS], c(1, 2), mean), thetat), 3)
round(apply(THETA[, , burnin:RS], c(1, 2), sd), 3)

#�P�ꕪ�z�̎��㕽��
round(cbind(t(apply(PHI1[, , burnin:RS], c(1, 2), mean)), t(phit1)), 3)
round(t(apply(PHI1[, , burnin:RS], c(1, 2), sd)), 3)
round(cbind(t(apply(PHI2[, , burnin:RS], c(1, 2), mean)), t(phit2)), 3)
round(t(apply(PHI2[, , burnin:RS], c(1, 2), sd)), 3)



##���ݕϐ��̃T���v�����O���ʂ̎��㕪�z
seg11_rate <- SEG11 / max(SEG11); seg12_rate <- SEG12 / max(SEG11)
seg21_rate <- SEG12 / max(SEG11)
seg2_rate <- SEG2 / rowSums(SEG2)
seg11_rate[is.nan(seg11_rate)] <- 0; seg12_rate[is.nan(seg12_rate)] <- 0
seg21_rate[is.nan(seg21_rate)] <- 0
seg2_rate[is.nan(seg2_rate)] <- 0

#�g�s�b�N�������ʂ��r
round(cbind(SEG11, seg11_rate, ZT1), 3)
round(cbind(rowSums(SEG12), seg12_rate), 3)
round(cbind(rowSums(SEG2), seg2_rate, Z2), 3)

