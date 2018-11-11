#####Tree_Structured_Switching_LDA_model#####
options(warn=0)
library(stringr)
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
s <- 3   #�f�[�^�^�C�v��
k0 <- 10   #��ʌ�̃g�s�b�N��
k11 <- 3   #���[�U�[�̏�ʃg�s�b�N��
k12 <- rtpois(k11, a=1, b=5, 2.5)   #���[�U�[�̉��ʃg�s�b�N�X
k21 <- 3   #�A�C�e���̏�ʃg�s�b�N��
k22 <- rtpois(k21, a=1, b=5, 2.5)   #�A�C�e���̉��ʃg�s�b�N��
hh <- 1000   #���r���A�[��
item <- 500   #�A�C�e����
v1 <- 300   #�]���X�R�A�̌�b��
v2 <- 400   #���[�U�[�g�s�b�N�̌�b��
v3 <- 400   #�A�C�e���g�s�b�N�̌�b��
v <- v1 + v2 + v3   #����b��
spl <- matrix(1:v1, nrow=s, ncol=v1/s, byrow=T)
v1_index <- 1:v1
v2_index <- (v1+1):(v1+v2)
v3_index <- (v2+1):v

##ID�ƌ����x�N�g���̍쐬
#ID�����ݒ�
user_id0 <- rep(1:hh, rep(item, hh))
item_id0 <- rep(1:item, hh)

#�����x�N�g�����쐬
for(rp in 1:100){
  m_vec <- rep(0, hh*item)
  for(i in 1:item){
    prob <- runif(1, 0.02, 0.12)
    m_vec[item_id0==i] <- rbinom(hh, 1, prob)
  }
  m_index <- which(m_vec==1)
  
  #���S��ID��ݒ�
  user_id <- user_id0[m_index]
  item_id <- item_id0[m_index]
  d <- length(user_id)   #�����r���[��
  
  #���ׂẴp�^�[��������������break
  if(length(unique(user_id))==hh & length(unique(item_id))==item) break
}

#�P�ꐔ��ݒ�
w <- rpois(d, rgamma(d, 25, 0.5))   #����������̒P�ꐔ
f <- sum(w)   #���P�ꐔ
n_user <- plyr::count(user_id)$freq
n_item <- plyr::count(item_id)$freq

#�P��ID��ݒ�
u_id <- rep(user_id, w)
i_id <- rep(item_id, w)
d_id <- rep(1:d, w)


#�C���f�b�N�X��ݒ�
user_index <- list()
item_index <- list()
for(i in 1:hh){
  user_index[[i]] <- which(user_id==i)
}
for(j in 1:item){
  item_index[[j]] <- which(item_id==j)
}


##�p�����[�^�̎��O���z��ݒ�
#�P�ꕪ�z�̎��O���z
alpha11 <- c(rep(0.1, v1), rep(0.001, v2+v3))
alpha12 <- c(rep(0.001, v1), rep(0.1, v2), rep(0.001, v3))
alpha13 <- c(rep(0.001, v1+v2), rep(0.1, v3))

#�X�C�b�`���O�ϐ��̎��O���z
alpha2 <- c(1.5, 3.0, 3.0)

#�؍\���g�s�b�N���z�̎��O���z
alpha31 <- c(22.5, 15.0)   #�ʉߊm���̎��O���z
alpha32 <- rep(0.2, max(c(k12, k22)))   #�m�[�h�I���m���̎��O���z

###���ׂĂ̒P�ꂪ�o������܂Ńf�[�^�̐����𑱂���
for(rp in 1:1000){
    
  ##�p�����[�^�𐶐�
  #�f�B���N�����z����P�ꕪ�z�𐶐�
  phi0 <- phit0 <- extraDistr::rdirichlet(k0, alpha11)
  phi11 <- phit11 <- extraDistr::rdirichlet(k11, alpha12)
  phi21 <- phit21 <- extraDistr::rdirichlet(k21, alpha13)
  phi12 <- phi22 <- list()
  for(j in 1:k11){
    phi12[[j]] <- extraDistr::rdirichlet(k12[j], alpha12)
  }
  for(j in 1:k21){
    phi22[[j]] <- extraDistr::rdirichlet(k22[j], alpha13)
  }
  phit12 <- phi12; phit22 <- phi22
  
  #�f�B���N�����z����X�C�b�`���O�ϐ��𐶐�
  lambda <- lambdat <- extraDistr::rdirichlet(hh, alpha2)
  
  ##�؍\���g�s�b�N�𐶐�
  #�x�[�^���z����ʉߊm���𐶐�
  gamma1 <- gammat1 <- matrix(rbeta(hh*k11, alpha31[1], alpha31[2]), nrow=hh, ncol=k11)   #���[�U�[�̒ʉߗ��𐶐�
  gamma2 <- gammat2 <- matrix(rbeta(item*k21, alpha31[1], alpha31[2]), nrow=item, ncol=k21)   #�A�C�e���̒ʉߗ��𐶐�
  
  #�f�B���N�����z����m�[�h�����m���𐶐�
  theta0 <- thetat0 <- as.numeric(extraDistr::rdirichlet(1, rep(2.5, k0)))
  theta11 <- thetat11 <- extraDistr::rdirichlet(hh, alpha32[1:k11])
  theta21 <- thetat21 <- extraDistr::rdirichlet(item, alpha32[1:k21])
  theta12 <- theta22 <- list()
  for(j in 1:k11){
    theta12[[j]] <- extraDistr::rdirichlet(hh, alpha32[1:k12[j]])
  }
  for(j in 1:k21){
    theta22[[j]] <- extraDistr::rdirichlet(item, alpha32[1:k22[j]])
  }
  thetat12 <- theta12; thetat22 <- theta22
  
  
  ##���f���Ɋ�Â��f�[�^�𐶐�
  #�f�[�^�̊i�[�p�z��
  WX <- matrix(0, nrow=d, ncol=v)
  wd_list <- y_list <- list()
  Z0_list <- Z1_list <- Z2_list <- G_list <- list() 
  options(warn=2)
  for(i in 1:d){
    if(i%%1000==0){
      print(i)
    }
    ##���[�U�[�ƃA�C�e���𒊏o
    word <- matrix(0, nrow=w[i], ncol=v)
    u_index <- user_id[i]
    i_index <- item_id[i]
    
    ##�������z����X�C�b�`���O�ϐ��𐶐�
    y <- rmnom(w[i], 1, lambda[u_index, ])
    y_vec <- as.numeric(y %*% 1:s)
    index_y1 <- which(y[, 1]==1); index_y2 <- which(y[, 2]==1); index_y3 <- which(y[, 3]==1)
    
    ##��ʌ�̃g�s�b�N�ƒP��𐶐�
    z0 <- matrix(0, nrow=w[i], ncol=k0)
    if(length(index_y1) > 0){
      #�g�s�b�N�𐶐�
      z0[index_y1, ] <- rmnom(length(index_y1), 1, theta0)
      z0_vec <- as.numeric(z0 %*% 1:k0)
      
      #�P��𐶐�
      word[index_y1, ] <- rmnom(length(index_y1), 1, phi0[z0_vec[index_y1], ])
    }
    
    ##���[�U�[�g�s�b�N�𐶐�
    #�f�[�^�̊i�[�p�z��
    z1 <- matrix(0, nrow=w[i], ncol=2)
    g1 <- rep(0, w[i])
    
    if(length(index_y2) > 0){
  
      #��ʊK�w�̃g�s�b�N�𐶐�
      z11 <- rmnom(length(index_y2), 1, theta11[u_index, ])
      z11_vec <- as.numeric(z11 %*% 1:k11)
      z1[index_y2, 1] <- z11_vec
      
      #�ʉߕϐ��𐶐�
      g1[index_y2] <- rbinom(length(index_y2), 1, gamma1[u_index, ][z1[, 1]])
      
      #���ʊK�w�̃g�s�b�N�𐶐�
      index_g1 <- which(g1==1)
      if(length(index_g1) > 0){
        for(j in 1:length(index_g1)){
          z12 <- as.numeric(rmnom(1, 1, theta12[[z1[index_g1, 1][j]]][u_index, ]))
          z1[index_g1[j], 2] <- as.numeric(z12 %*% 1:length(z12))
        }
      }
      #�g�s�b�N����P��𐶐�
      for(j in 1:length(index_y2)){
        node <- z1[index_y2[j], ]
        if(sum(node > 0)==1){
          word[index_y2[j], ] <- rmnom(1, 1, phi11[node[1], ])
        } else {
          word[index_y2[j], ] <- rmnom(1, 1, phi12[[node[1]]][node[2], ])
        }
      }
    }
    
    ##�A�C�e���g�s�b�N�𐶐�
    #�f�[�^�̊i�[�p�z��
    z2 <- matrix(0, nrow=w[i], ncol=2)
    g2 <- rep(0, w[i])
    
    if(length(index_y3) > 0){
      #��ʊK�w�̃g�s�b�N�𐶐�
      z21 <- rmnom(length(index_y3), 1, theta21[i_index, ])
      z21_vec <- as.numeric(z21 %*% 1:k21)
      z2[index_y3, 1] <- z21_vec
      
      #�ʉߕϐ��𐶐�
      g2[index_y3] <- rbinom(length(index_y3), 1, gamma2[i_index, ][z2[, 1]])
      
      #���ʊK�w�̃g�s�b�N�𐶐�
      index_g2 <- which(g2==1)
      if(length(index_g2) > 0){
        for(j in 1:length(index_g2)){
          z22 <- as.numeric(rmnom(1, 1, theta22[[z2[index_g2, 1][j]]][i_index, ]))
          z2[index_g2[j], 2] <- as.numeric(z22 %*% 1:length(z22))
        }
      }
      #�g�s�b�N����P��𐶐�
      for(j in 1:length(index_y3)){
        node <- z2[index_y3[j], ]
        if(sum(node > 0)==1){
          word[index_y3[j], ] <- rmnom(1, 1, phi21[node[1], ])
        } else {
          word[index_y3[j], ] <- rmnom(1, 1, phi22[[node[1]]][node[2], ])
        }
      }
    }
    
    ##�f�[�^���i�[
    WX[i, ] <- colSums(word)
    wd_list[[i]] <- as.numeric(word %*% 1:v)
    y_list[[i]] <- y
    Z0_list[[i]] <- as.numeric(z0 %*% 1:k0)
    Z1_list[[i]] <- z1
    Z2_list[[i]] <- z2
    G_list[[i]] <- cbind(g1, g2)
  }
  if(min(colSums(WX)) > 0) break   #break����
}

##���X�g��ϊ�
wd <- unlist(wd_list)
y <- do.call(rbind, y_list)
Z0 <- unlist(Z0_list)
Z1 <- do.call(rbind, Z1_list)
Z2 <- do.call(rbind, Z2_list)
G <- do.call(rbind, G_list)
sparse_data <- sparseMatrix(i=1:f, wd, x=rep(1, f), dims=c(f, v))
sparse_data_T <- t(sparse_data)
rm(wd_list); rm(y_list); rm(Z0_list); rm(Z1_list); rm(Z2_list); rm(G_list)
gc(); gc()


####�}���R�t�A�������e�J�����@��Tree Structured Switching LDA model�𐄒�####
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
burnin <- 1000
disp <- 10

##���O���z�̐ݒ�
alpha00 <- 10.0
alpha01 <- 0.1
alpha02 <- 0.1
beta01 <- 1.0
beta02 <- 1.0

##�p�����[�^�̐^�l
lambda <- lambdat
gamma1 <- gammat1
gamma2 <- gammat2
theta0 <- thetat0
theta11 <- thetat11
theta12 <- thetat12
theta21 <- thetat21
theta22 <- thetat22
phi0 <- phit0
phi11 <- phit11
phi12 <- phit12
phi21 <- phit21
phi22 <- phit22

##�����l�̐ݒ�
lambda <- extraDistr::rdirichlet(hh, rep(10.0, s))
gamma1 <- matrix(0.5, nrow=hh, ncol=k11)
gamma2 <- matrix(0.5, nrow=item, ncol=k21)
theta0 <- as.numeric(extraDistr::rdirichlet(1, rep(10.0, k0)))
theta11 <- extraDistr::rdirichlet(hh, rep(5.0, k11))
theta21 <- extraDistr::rdirichlet(item, rep(5.0, k21))
phi0 <- extraDistr::rdirichlet(k0, rep(5.0, v))
phi11 <- extraDistr::rdirichlet(k11, rep(5.0, v))
phi21 <- extraDistr::rdirichlet(k21, rep(5.0, v))
theta12 <- theta22 <- list()
phi12 <- phi22 <- list()
for(j in 1:k11){
  theta12[[j]] <- extraDistr::rdirichlet(hh, rep(10.0, k12[j]))
  phi12[[j]] <- extraDistr::rdirichlet(k12[j], rep(10.0, v))
}
for(j in 1:k21){
  theta22[[j]] <- extraDistr::rdirichlet(item, rep(10.0, k22[j]))
  phi22[[j]] <- extraDistr::rdirichlet(k22[j], rep(5.0, v))
}

##�p�����[�^�̊i�[�p�z��

##�p�����[�^�Ɛ��ݕϐ��̊i�[�p�z��
#�p�����[�^�̊i�[�p�z��
LAMBDA <- array(0, dim=c(hh, s, R/keep))
GAMMA1 <- array(0, dim=c(hh, k11, R/keep))
GAMMA2 <- array(0, dim=c(item, k21, R/keep))
THETA0 <- matrix(0, nrow=R/keep, ncol=k0)
THETA11 <- array(0, dim=c(hh, k11, R/keep))
THETA12 <- array(0, dim=c(hh, sum(k12), R/keep))
THETA21 <- array(0, dim=c(item, k21, R/keep))
THETA22 <- array(0, dim=c(item, sum(k22), R/keep))
PHI0 <- array(0, dim=c(k0, v, R/keep))
PHI11 <- array(0, dim=c(k11, v, R/keep))
PHI12 <- array(0, dim=c(sum(k12), v, R/keep))
PHI21 <- array(0, dim=c(k21, v, R/keep))
PHI22 <- array(0, dim=c(sum(k22), v, R/keep))

#���ݕϐ��̊i�[�p�z��
Si <- matrix(0, nrow=f, ncol=s)
SEG0 <- matrix(0, nrow=f, ncol=k0)
SEG11 <- matrix(0, nrow=f, ncol=k11)
SEG12 <- matrix(0, nrow=f, ncol=sum(k12))
SEG21 <- matrix(0, nrow=f, ncol=k21)
SEG22 <- matrix(0, nrow=f, ncol=sum(k22))

##�f�[�^�̃C���f�b�N�X��ݒ�
#�C���f�b�N�X��ݒ�
user_list <- user_vec <- list()
item_list <- item_vec <- list()
word_list <- word_vec <- list()
for(i in 1:hh){
  user_list[[i]] <- which(u_id==i)
  user_vec[[i]] <- rep(1, length(user_list[[i]]))
}
for(j in 1:item){
  item_list[[j]] <- which(i_id==j)
  item_vec[[j]] <- rep(1, length(item_list[[j]]))
}
for(j in 1:v){
  word_list[[j]] <- which(wd==j)
  word_vec[[j]] <- rep(1, length(word_list[[j]]))
}
vec0 <- rep(1, k0); vec11 <- rep(1, k11); vec21 <- rep(1, k21)

#�ΐ��ޓx�̊�l
LLst <- sum(sparse_data %*% log(colSums(sparse_data) / f))


####�M�u�X�T���v�����O�Ńp�����[�^���T���v�����O####
for(rp in 1:R){
  
  ##�p�����[�^�̊i�[�p�z��̐ݒ�
  wsum012 <- vsum012 <- s01 <- list()
  wsum022 <- vsum022 <- s02 <- list()
  for(j in 1:k11){
    wsum012[[j]] <- matrix(0, nrow=hh, ncol=k12[j])
    s01[[j]] <- matrix(0, nrow=hh, ncol=2)
    vsum012[[j]] <- matrix(0, nrow=k12[j], ncol=v)
  }
  for(j in 1:k21){
    wsum022[[j]] <- matrix(0, nrow=hh, ncol=k22[j])
    s02[[j]] <- matrix(0, nrow=item, ncol=2)
    vsum022[[j]] <- matrix(0, nrow=k22[j], ncol=v)
  }

  ##���ґΐ��ޓx��ݒ�
  #��ʌ�̊��ґΐ��ޓx
  Lho0 <- matrix(theta0, nrow=f, ncol=k0, byrow=T) * t(phi0)[wd, ]  
  Lis0 <- as.numeric(Lho0 %*% vec0)
  
  #���[�U�[�֘A��̊��ґΐ��ޓx
  Lho11 <- theta11[u_id, ] * t(phi11)[wd, ]
  Lis11 <- as.numeric(((1-gamma1[u_id, ]) * Lho11) %*% vec11)
  Lho12 <- list(); Lis12 <- matrix(0, nrow=f, ncol=k11)
  for(j in 1:k11){
    Lho12[[j]] <- theta12[[j]][u_id, ] * t(phi12[[j]])[wd, ]
    Lis12[, j] <- gamma1[u_id, j] * as.numeric(Lho12[[j]] %*% rep(1, k12[j]))
  }
  
  #�A�C�e���֘A��̊��ґΐ��ޓx
  Lho21 <- theta21[i_id, ] * t(phi21)[wd, ]
  Lis21 <- as.numeric(((1-gamma2[i_id, ]) * Lho21) %*% vec21)
  Lho22 <- list(); Lis22 <- matrix(0, nrow=f, ncol=k21)
  for(j in 1:k21){
    Lho22[[j]] <- theta22[[j]][i_id, ] * t(phi22[[j]])[wd, ]
    Lis22[, j] <- gamma2[i_id, j] * as.numeric(Lho22[[j]] %*% rep(1, k22[j]))
  }
  
  ##�������z����X�C�b�`���O�ϐ����T���v�����O
  #�X�C�b�`���O�ϐ��̊����m��
  Lis12_1 <- theta11[u_id, ] * Lis12
  Lis22_1 <- theta21[i_id, ] * Lis22
  Lis1 <- Lis11 + as.numeric(Lis12_1 %*% vec11)
  Lis2 <- Lis21 + as.numeric(Lis22_1 %*% vec21)
  Lis_par <- lambda[u_id, ] * cbind(Lis0, Lis1, Lis2)
  y_rate <- Lis_par / as.numeric(Lis_par %*% rep(1, s))   #���ݕϐ��̊����m��
  
  #�X�C�b�`���O�ϐ����T���v�����O
  yi <- rmnom(f, 1, y_rate)   #�������z����T���v�����O
  yi_T <- t(yi)
  y_vec <- as.numeric(yi %*% 1:s)
  index_general <- which(yi[, 1]==1); index_user <- which(yi[, 2]==1); index_item <- which(yi[, 3]==1)
  
  
  ##���[�U�[�g�s�b�N�̖؍\���𐶐�
  ##��ʊK�w�̃m�[�h�𐶐�
  #�g�s�b�N�m�[�h�̊����m��
  par <- ((1-gamma1[u_id[index_user], ]) * Lho11[index_user, ]) + Lis12_1[index_user, ]
  par_rate <- par / as.numeric(par %*% vec11)   #���ݕϐ��̊����m��
  
  #�������z����g�s�b�N�m�[�h�𐶐�
  Zi11 <- matrix(0, nrow=f, ncol=k11)
  Zi11[index_user, ] <- rmnom(length(index_user), 1, par_rate)
  z11_vec <- as.numeric(Zi11 %*% 1:k11)
  Zi11_T <- t(Zi11)
  
  ##�ʉߕϐ����T���v�����O
  #�ʉߊm����ݒ�
  g1_rate <- Lis12_1[index_user, ] / par
  g1_rate[is.nan(g1_rate)] <- 0
  
  #�x���k�[�C���z����ʉߕϐ����T���v�����O
  g1_data <- matrix(0, nrow=f, ncol=k11)
  for(j in 1:k11){
    g1_data[index_user, j] <- rbinom(length(index_user), 1, as.numeric((g1_rate[, j] * Zi11[index_user, j])))
  }
  g1 <- as.numeric(g1_data %*% vec11)
  
  ##���ʊK�w�̃m�[�h�𐶐�
  Zi12_T_list <- list()
  z12_vec <- matrix(0, nrow=f, ncol=k11)
  
  for(j in 1:k11){
    #�g�s�b�N�m�[�h�̊����m��
    index_node <- which(Zi11[, j]==1 & g1==1)
    par <- Lho12[[j]][index_node, ]
    par_rate <- par / as.numeric(par %*% rep(1, ncol(par)))   #���ݕϐ��̊����m��
    
    #�������z����g�s�b�N�m�[�h�𐶐�
    Zi12 <- matrix(0, nrow=f, ncol=k12[j])
    Zi12[index_node, ] <- rmnom(length(index_node), 1, par_rate)  
    Zi12_T_list[[j]] <- t(Zi12)
    z12_vec[, j] <- as.numeric(Zi12 %*% 1:k12[j])
  }
  
  
  ##�A�C�e���g�s�b�N�̖؍\���𐶐�
  ##��ʊK�w�̃m�[�h�𐶐�
  #�g�s�b�N�m�[�h�̊����m��
  par <- ((1-gamma2[i_id[index_item], ]) * Lho21[index_item, ]) + Lis22_1[index_item, ]
  par_rate <- par / as.numeric(par %*% vec21)   #���ݕϐ��̊����m��
  
  #�������z����g�s�b�N�m�[�h�𐶐�
  Zi21 <- matrix(0, nrow=f, ncol=k21)
  Zi21[index_item, ] <- rmnom(length(index_item), 1, par_rate)
  z21_vec <- as.numeric(Zi21 %*% 1:k21)
  Zi21_T <- t(Zi21)
  
  ##�ʉߕϐ����T���v�����O
  #�ʉߊm����ݒ�
  g2_rate <- Lis22_1[index_item, ] / par
  g2_rate[is.nan(g2_rate)] <- 0
  
  #�x���k�[�C���z����ʉߕϐ����T���v�����O
  g2_data <- matrix(0, nrow=f, ncol=k21)
  for(j in 1:k21){
    g2_data[index_item, j] <- rbinom(length(index_item), 1, as.numeric((g2_rate[, j] * Zi21[index_item, j])))
  }
  g2 <- as.numeric(g2_data %*% vec21)
  
  ##���ʊK�w�̃m�[�h�𐶐�
  Zi22_T_list <- list()
  z22_vec <- matrix(0, nrow=f, ncol=k21)
  
  for(j in 1:k21){
    #�g�s�b�N�m�[�h�̊����m��
    index_node <- which(Zi21[, j]==1 & g2==1)
    par <- Lho22[[j]][index_node, ]
    par_rate <- par / as.numeric(par %*% rep(1, ncol(par)))   #���ݕϐ��̊����m��
    
    #�������z����g�s�b�N�m�[�h�𐶐�
    Zi22 <- matrix(0, nrow=f, ncol=k22[j])
    Zi22[index_node, ] <- rmnom(length(index_node), 1, par_rate)  
    Zi22_T_list[[j]] <- t(Zi22)
    z22_vec[, j] <- as.numeric(Zi22 %*% 1:k22[j])
  }
  
  ##��ʌ�̃g�s�b�N���T���v�����O
  #�g�s�b�N�̊����m��
  par <- Lho0[index_general, ]
  par_rate <- par / as.numeric(par %*% vec0)   #���ݕϐ��̊����m��
  
  #�������z����g�s�b�N�𐶐�
  Zi0 <- matrix(0, nrow=f, ncol=k0)
  Zi0[index_general, ] <- rmnom(length(index_general), 1, par_rate)
  
  ##�X�C�b�`���O�ϐ��̍��������T���v�����O
  rsum0 <- matrix(0, nrow=hh, ncol=s)
  for(i in 1:hh){
    rsum0[i, ] <- yi_T[, user_list[[i]]] %*% user_vec[[i]]
  }
  rsum <- rsum0 + beta01   #�f�B���N�����z�̃p�����[�^
  lambda <- extraDistr::rdirichlet(hh, rsum)   #�p�����[�^���T���v�����O
  
  ##��ʌ�̃g�s�b�N���z���T���v�����O
  wsum <- colSums(Zi0) + beta01
  theta0 <- as.numeric(extraDistr::rdirichlet(1, wsum))
  
  ##���[�U�[�̃g�s�b�N���z����ђʉߗ����T���v�����O
  #�f�B���N�����z����уx�[�^���z�̃p�����[�^
  wsum011 <- matrix(0, nrow=hh, ncol=k11)
  s01 <- array(0, dim=c(hh, 2, k11))
  for(i in 1:hh){
    wsum011[i, ] <- Zi11_T[, user_list[[i]]] %*% user_vec[[i]]
    for(j in 1:k11){
      s01[i, 1, j] <- sum(Zi11[user_list[[i]], j] * g1[user_list[[i]]])
      s01[i, 2, j] <- sum(Zi11[user_list[[i]], j])
      wsum012[[j]][i, ] <- Zi12_T_list[[j]][, user_list[[i]]] %*% user_vec[[i]]
    }
  }
  #�g�s�b�N���z�ƒʉߗ����T���v�����O
  theta11 <- extraDistr::rdirichlet(hh, wsum011 + alpha01)
  for(j in 1:k11){
    theta12[[j]] <- extraDistr::rdirichlet(hh, wsum012[[j]] + alpha01)
    gamma1[, j] <- rbeta(hh, s01[, 1, j]+beta01, s01[, 2, j]-s01[, 1, j]+beta01)
  }
  
  ##�A�C�e���̃g�s�b�N���z����ђʉߗ����T���v�����O
  #�f�B���N�����z����уx�[�^���z�̃p�����[�^
  wsum021 <- matrix(0, nrow=item, ncol=k21)
  s02 <- array(0, dim=c(hh, 2, k21))
  for(i in 1:item){
    wsum021[i, ] <- Zi21_T[, item_list[[i]]] %*% item_vec[[i]]
    for(j in 1:k21){
      s02[i, 1, j] <- sum(Zi21[item_list[[i]], j] * g2[item_list[[i]]])
      s02[i, 2, j] <- sum(Zi21[item_list[[i]], j])
      wsum022[[j]][i, ] <- Zi22_T_list[[j]][, item_list[[i]]] %*% item_vec[[i]]
    }
  }
  #�g�s�b�N���z�ƒʉߗ����T���v�����O
  theta21 <- extraDistr::rdirichlet(item, wsum021 + alpha01)
  for(j in 1:k21){
    theta22[[j]] <- extraDistr::rdirichlet(item, wsum022[[j]] + alpha01)
    gamma2[, j] <- rbeta(item, s02[, 1, j]+beta01, s02[, 2, j]-s02[, 1, j] + beta01)
  }
  
  
  ##�P�ꕪ�z���T���v�����O
  #�f�B���N�����z�̃p�����[�^
  vsum011 <- matrix(0, nrow=k11, ncol=v)
  vsum021 <- matrix(0, nrow=k21, ncol=v)
  for(i in 1:v){
    #��ʊK�w�̒P�ꕪ�z�̃p�����[�^
    r11 <- matrix(1-g1[word_list[[i]]], nrow=k11, ncol=length(word_vec[[i]]), byrow=T)
    r21 <- matrix(1-g2[word_list[[i]]], nrow=k21, ncol=length(word_vec[[i]]), byrow=T)
    vsum011[, i] <- (Zi11_T[, word_list[[i]]] * r11) %*% word_vec[[i]]
    vsum021[, i] <- (Zi21_T[, word_list[[i]]] * r21) %*% word_vec[[i]]
  
    #���ʊK�w�̒P�ꕪ�z�̃p�����[�^
    for(j1 in 1:k11){
      vsum012[[j1]][, i] <- Zi12_T_list[[j1]][, word_list[[i]], drop=FALSE] %*% word_vec[[i]]
    }
    for(j2 in 1:k21){
      vsum022[[j2]][, i] <- Zi22_T_list[[j2]][, word_list[[i]], drop=FALSE] %*% word_vec[[i]]
    }
  }
  vsum0 <- as.matrix(t(sparse_data_T %*% Zi0))
  
  #�f�B���N�����z����p�����[�^���T���v�����O
  #��ʊK�w�̒P�ꕪ�z���T���v�����O
  phi0 <- extraDistr::rdirichlet(k0, vsum0 + alpha02)
  phi11 <- extraDistr::rdirichlet(k11, vsum011 + alpha02)
  phi21 <- extraDistr::rdirichlet(k21, vsum021 + alpha02)
  
  #���ʊK�w�̒P�ꕪ�z���T���v�����O
  for(j in 1:k11){
    phi12[[j]] <- extraDistr::rdirichlet(k12[j], vsum012[[j]] + alpha02)
  }
  for(j in 1:k21){
    phi22[[j]] <- extraDistr::rdirichlet(k22[j], vsum022[[j]] + alpha02)
  }
  
  ##�p�����[�^�̊i�[�ƃT���v�����O���ʂ̕\��
  #�T���v�����O���ꂽ�p�����[�^
  if(rp%%keep==0){
    #�T���v�����O���ʂ̊i�[
    mkeep <- rp/keep
    LAMBDA[, , mkeep] <- lambda
    GAMMA1[, , mkeep] <- gamma1
    GAMMA2[, , mkeep] <- gamma2
    THETA0[mkeep, ] <- theta0
    THETA11[, , mkeep] <- theta11
    THETA12[, , mkeep] <- matrix(unlist(theta12), nrow=hh, ncol=sum(k12))
    THETA21[, , mkeep] <- theta21
    THETA22[, , mkeep] <- matrix(unlist(theta22), nrow=item, ncol=sum(k22))
    PHI0[, , mkeep] <- phi0
    PHI11[, , mkeep] <- phi11
    PHI12[, , mkeep] <- do.call(rbind, phi12)
    PHI21[, , mkeep] <- phi21
    PHI22[, , mkeep] <- do.call(rbind, phi22)
  }  
  
  #�g�s�b�N�����̓o�[���C�����Ԃ𒴂�����i�[����
  if(rp%%keep==0 & rp >= burnin){
    Si <- Si + yi
    SEG0 <- SEG0 + Zi0
    SEG11 <- SEG11 + Zi11
    SEG12 <- SEG12 + t(do.call(rbind, Zi12_T_list))
    SEG21 <- SEG21 + Zi21
    SEG22 <- SEG22 + t(do.call(rbind, Zi22_T_list))
  }

  ##�ΐ��ޓx�ƃT���v�����O���ʂ̕\��
  if(rp%%disp==0){
    #�ΐ��ޓx�̌v�Z
    LL12_vec <- LL22_vec <- c()
    Lho00 <- rowSums(Lho0 * yi[, 1]); LL0 <- sum(log(Lho00[Lho00!=0]))
    Lho011 <- rowSums(Lho11[index_user, ] * (1-rowSums(g1_data[index_user, ]))); LL11 <- sum(log(Lho011[Lho011!=0]))
    Lho021 <- rowSums(Lho21[index_item, ] * (1-rowSums(g2_data[index_item, ]))); LL21 <- sum(log(Lho021[Lho021!=0]))
    for(j in 1:k11){
      LL12_vec <- c(LL12_vec, sum(log(rowSums(Lho12[[j]][rowSums(t(Zi12_T_list[[j]]))==1, ]))))
    }
    for(j in 1:k21){
      LL22_vec <- c(LL22_vec, sum(log(rowSums(Lho22[[j]][rowSums(t(Zi22_T_list[[j]]))==1, ]))))
    }
    LL12 <- sum(LL12_vec); LL22 <- sum(LL22_vec)
    LL <- LL0 + LL11 + LL12 + LL21 + LL22   #�ΐ��ޓx�̘a
    
    #�T���v�����O���ʂ̕\��
    print(rp)
    print(c(LL, LLst))
    print(round(rbind(theta0, thetat0), 3))
    print(round(cbind(lambda, lambdat)[1:10, ], 3))
  }
}

