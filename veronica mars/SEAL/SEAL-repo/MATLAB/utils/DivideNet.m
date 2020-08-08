function [train test] = DivideNet(net, ratioTrain, connected)
%%����ѵ�����Ͳ��Լ�����֤ѵ������ͨ
    net = triu(net) - diag(diag(net));  % convert to upper triangular matrix
    num_testlinks = floor((1-ratioTrain) * nnz(net));      
    % ȷ�����Լ��ı���Ŀ
    [xindex, yindex] = find(net);  linklist = [xindex yindex];    
    % �����磨�ڽӾ��������еı��ҳ���������linklist  
    clear xindex yindex;  
    % Ϊÿ�������ñ�־λ���ж��Ƿ���ɾ��
    test = sparse(size(net,1),size(net,2));                 
    while (nnz(test) < num_testlinks)               %For power dataset, maximum 636 test links <660 expected. 
        if length(linklist) <= 2
            break;
        end
        %---- ���ѡ��һ����
        index_link = ceil(rand(1) * length(linklist));
        
        uid1 = linklist(index_link,1); 
        uid2 = linklist(index_link,2);    
        net(uid1,uid2) = 0;  
        
        
          %% 
        %---- �ж���ѡ�����˽ڵ�uid1��uid2�Ƿ�ɴ���ɴ���ɷ�����Լ�������������ѡһ����
         
        % �������ߴ���������ȥ�����ж��ڵ���������Ƿ���ͨ
        tempvector = net(uid1,:);
        % ȡ��uid1һ���ɴ�ĵ㣬������һά����
        sign = 0;  
        % ��Ǵ˱��Ƿ���Ա��Ƴ���sign=0��ʾ���ɣ� sign=1��ʾ����
        uid1TOuid2 = tempvector * net + tempvector;        
        % uid1TOuid2��ʾ�����ڿɴ�ĵ�
        if uid1TOuid2(uid2) > 0
            sign = 1;               
            % �������ɴ�
        else
            while (nnz(spones(uid1TOuid2) - tempvector) ~=0)   
            % ֱ���ɴ�ĵ㵽���ȶ�״̬����Ȼ���ܵ���uid2���˱߾Ͳ��ܱ�ɾ��
                tempvector = spones(uid1TOuid2);
                uid1TOuid2 = tempvector * net + tempvector;    
                % �˲���uid1TOuid2��ʾK���ڿɴ�ĵ�
                if uid1TOuid2(uid2) > 0
                    sign = 1;      
                     % ĳ���ڿɴ�
                    break;
                end
            end
        end 
        % ����-�ж�uid1�Ƿ�ɴ�uid2
        
        if connected == false
            sign = 1;  % overwrite, keep all selected links in test, no matter whether the remaining net is connected
        end

        %% 
        
        %----���˱߿�ɾ������֮������Լ��У������˱ߴ�linklist���Ƴ�
        if sign == 1 %�˱߿���ɾ��
            linklist(index_link,:) = []; 
            test(uid1,uid2) = 1;
        else
            linklist(index_link,:) = [];
            net(uid1,uid2) = 1;   
        end   
        % ����-�жϴ˱��Ƿ����ɾ��������Ӧ����
    end   
    % ������while��-���Լ��еı�ѡȡ���
    train = net + net';  test = test + test';
    % ����Ϊѵ�����Ͳ��Լ�
end
