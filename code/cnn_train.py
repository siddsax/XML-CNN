from header import *
from cnn_test import *

# ---------------------------------------------------------------------------------

def train(x_tr, y_tr, x_te, y_te, embedding_weights, params):
		
	viz = Visdom()
	loss_best = float('Inf')
	bestTotalLoss = float('Inf')
	best_test_acc = 0

	num_mb = np.ceil(params.N/params.mb_size)
	
	model = xmlCNN(params, embedding_weights)
	if(torch.cuda.is_available()):
		print("--------------- Using GPU! ---------")
		model.params.dtype_f = torch.cuda.FloatTensor
		model.params.dtype_i = torch.cuda.LongTensor
		
		model = model.cuda()
	else:
		model.params.dtype_f = torch.FloatTensor
		model.params.dtype_i = torch.LongTensor
		print("=============== Using CPU =========")

	optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)
	print(model);print("%"*100)
	
	if params.dataparallel:
		model = nn.DataParallel(model)
	
	if(len(params.load_model)):
		params.model_name = params.load_model
		print(params.load_model)
		model, optimizer, init = load_model(model, params.load_model, optimizer=optimizer)
	else:
		init = 0
	iteration = 0
	# =============================== TRAINING ====================================
	for epoch in range(init, params.num_epochs):
		totalLoss = 0.0

		for i in range(int(num_mb)):
			# ------------------ Load Batch Data ---------------------------------------------------------
			batch_x, batch_y = load_batch_cnn(x_tr, y_tr, params)
			# -----------------------------------------------------------------------------------
			loss, output = model.forward(batch_x, batch_y)
			loss = loss.mean().squeeze()
			# --------------------------------------------------------------------

			totalLoss += loss.data
			
			if i % int(num_mb/12) == 0:
				print('Iter-{}; Loss: {:.4}; best_loss: {:.4}'.format(i, loss.data, loss_best))
				if not os.path.exists('saved_models/' + params.model_name ):
					os.makedirs('saved_models/' + params.model_name)
				save_model(model, optimizer, epoch, params.model_name + "/model_best_batch")
				if(loss<loss_best):
					loss_best = loss.data

			# ------------------------ Propogate loss -----------------------------------
			loss.backward()
			loss = loss.data
			torch.nn.utils.clip_grad_norm(model.parameters(), params.clip)
			optimizer.step()
			optimizer.zero_grad()

			# ----------------------------------------------------------------------------
			if(params.disp_flg):
				if(iteration==0):
					loss_old = loss
				else:
					viz.line(X=np.linspace(iteration-1,iteration,50), Y=np.linspace(loss_old, loss,50), update='append', win=win)
					loss_old = loss
				if(iteration % 100 == 0 ):
					win = viz.line(X=np.arange(iteration, iteration + .1), Y=np.arange(0, .1))
			iteration +=1

			if(epoch==0):
				break

		if(totalLoss<bestTotalLoss):

			bestTotalLoss = totalLoss
			if not os.path.exists('saved_models/' + params.model_name ):
				os.makedirs('saved_models/' + params.model_name)
			save_model(model, optimizer, epoch, params.model_name + "/model_best_epoch")

		print('End-of-Epoch: Loss: {:.4}; best_loss: {:.4};'.format(totalLoss, bestTotalLoss))
	
		test_prec_acc, test_ce_loss = test_class(x_te, y_te, params, model=model, verbose=False, save=False)
		model.train()
		
		if(test_prec_acc > best_test_acc):
			best_test_loss = test_ce_loss
			best_test_acc = test_prec_acc
			print("This acc is better than the previous recored test acc:- {} ; while CELoss:- {}".format(best_test_acc, best_test_loss))
			if not os.path.exists('saved_models/' + params.model_name ):
				os.makedirs('saved_models/' + params.model_name)
			save_model(model, optimizer, epoch, params.model_name + "/model_best_test")

		if epoch % params.save_step == 0:
			save_model(model, optimizer, epoch, params.model_name + "/model_" + str(epoch))



