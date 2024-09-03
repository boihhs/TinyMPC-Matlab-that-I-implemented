%Leo Benaharon
%read/understood paper and coded in one day on 9/1/2024
%referenced https://arxiv.org/pdf/2403.18149

classdef TinyMPC
    properties
        %primal varibles
        X %updated during foward pass in primal update: X(k+1) = AX(k) + BU(k) + c; (X -> X+)
        U %updated during backward pass in primal update: U(k) = -KX(k) - d(k); (U -> U+)
        %slack varibles
        Z %updated in slack update by first setting Z = X+ and then projecting it onto the second order cone (II(z)); (Z -> Z+)
        W %updated in slack update by first setting W = U+ and then projecting it onto the second order cone (II(z)); (W -> W+)

        %dual varibles
        Lamda %updated in dual update by doing Lamda(k) = Lamda(k) + penalty(X(k)+ - Z(k)+); (Lamda -> Lamda+)
        Mu %updated in dual update by doing Mu(k) = Mu(k) + penalty(U(k)+ - W(k)+); (Mu -> Mu+)
        penalty %updated by doing penalty = 10*penalty; (penalty -> penalty+)

        %dynamics
        A %is given
        B %is given
        c %will probably set 0

        %cost function
        Q %is given
        R %is given
        q %will probably set 0
        r %will probably set 0

        %Augmented cost function (actually used in Riccati recursion)
        qhat %updated during backward pass in primal update before running Riccati recursion: qhat(k) = q(k) + Lamda(k) - penalty*Z(k)
        rhat %updated during backward pass in primal update before running Riccati recursion: rhat(k) = r(k) + Mu(k) - penalty*W(k)

        %Riccati recursion variables
        K %precomputed after solving infinite horizion LQR with just regular cost function (I think)
        P %precomputed after solving infinite horizion LQR with just regular cost function (I think)
        C1 %precomputed after solving infinite horizion LQR with just regular cost function (I think): C1 = (R + B.T*P(inf)*B)^-1
        C2 %precomputed after solving infinite horizion LQR with just regular cost function (I think): C2 = (A - B*K(inf)).T
        C3 %precomputed after solving infinite horizion LQR with just regular cost function (I think): C3 = B.T*P(inf)*c
        C4 %precomputed after solving infinite horizion LQR with just regular cost function (I think): C4 = C2*P(inf)*c
        d %updated during backward pass in primal update: d(k) = C1(B.T*p(k+1) + rhat(k) + C3)
        p %updated during backward pass in primal update: p(k) = qhat(k) + C2*p(k+1) - K(inf).T*rhat(k) + C4

        %Second Order Cone Constraints
        xCon %used in II() and is the magnitude that X(k) has to stay in
        uMax %used in II() and is the magnitude that U(k) has to stay in
        phi  %used in II() and is the cone angle that U(k) has to stay in
        

        tol %tolerance
        timesteps %seconds/dt

    end
    
    methods
        %initizie everything
        function obj = TinyMPC(X, U, A, B, c, Q, R, q, r, xCon, uMax, tol, timesteps, phi, penalty)
            obj.X = X;
            obj.U = U;
            obj.A = A;
            obj.B = B;
            obj.c = c;
            obj.Q = Q;
            obj.R = R;
            obj.q = q;
            obj.r = r;
            obj.xCon = xCon;
            obj.uMax = uMax;
            obj.tol = tol;
            obj.timesteps = timesteps;
            obj.Z = obj.X;
            obj.W = obj.U;
            obj.d = obj.U;
            obj.Lamda = zeros(size(obj.X));
            obj.Mu = obj.U;
            obj.phi = deg2rad(phi);
            obj.penalty = penalty;
           
        end

        %primal update
        %runs LQR, if aug is true, runs to solve backward pass in primal
        %update (updates d, p, then U and X). If aug is false, then runs LQR with regular cost function
        %to get K(inf), P(inf) and update C1, C2, C3, C4
        function obj = LQR(obj, aug)
            if aug
                obj.p = (obj.q + obj.Lamda(size(obj.Lamda, 1)) - obj.penalty*obj.Z(size(obj.Z, 1))).';
                for i = size(obj.U, 1):-1:1
                    obj.qhat = obj.q + obj.Lamda(size(obj.Lamda, 1)) - obj.penalty*obj.Z(size(obj.Z, 1));
                    obj.rhat = obj.r + obj.Mu(i, :) - obj.penalty*obj.W(i, :);

                    obj.d(i, :) = obj.C1*(obj.B.'*obj.p + obj.rhat.' + obj.C3);
                    obj.p = obj.qhat.' + obj.C2*obj.p - obj.K.'*obj.rhat.' + obj.C4;
                end

                for i=1:size(obj.U, 1)
                    obj.U(i, :) = -obj.K*obj.X(i, :).' - obj.d(i, :).';
                    obj.X(i+1, :) = obj.A*obj.X(i, :).' + obj.B*obj.U(i, :).' + obj.c.';
                end

            else

                obj.P = obj.Q;
                oldP = obj.P;
                p_ = obj.q.';

                obj.K = (obj.R + obj.B.'*obj.P*obj.B)^-1*(obj.B.'*obj.P*obj.A);
                d_ = (obj.R + obj.B.'*obj.P*obj.B)^-1*(obj.B.'*p_ + obj.r.' + obj.B.'*obj.P*obj.c.');
                obj.P = obj.Q + obj.K.'*obj.R*obj.K + (obj.A - obj.B*obj.K).'*obj.P*(obj.A - obj.B*obj.K);
                p_ = obj.q.' + (obj.A - obj.B*obj.K).'*(p_ - obj.P*obj.B*d_ + obj.P*obj.c.') + obj.K.'*(obj.R*d_ - obj.r.');

                while norm(obj.P - oldP) > obj.tol
                    oldP = obj.P;
                    obj.K = (obj.R + obj.B.'*obj.P*obj.B)^-1*(obj.B.'*obj.P*obj.A);
                    d_ = (obj.R + obj.B.'*obj.P*obj.B)^-1*(obj.B.'*p_ + obj.r.' + obj.B.'*obj.P*obj.c.');
                    obj.P = obj.Q + obj.K.'*obj.R*obj.K + (obj.A - obj.B*obj.K).'*obj.P*(obj.A - obj.B*obj.K);
                    p_ = obj.q.' + (obj.A - obj.B*obj.K).'*(p_ - obj.P*obj.B*d_ + obj.P*obj.c.') + obj.K.'*(obj.R*d_ - obj.r.');
                end
    
                obj.C1 = (obj.R + obj.B.'*obj.P*obj.B)^-1;
                obj.C2 = (obj.A - obj.B*obj.K).';
                obj.C3 = obj.B.'*obj.P*obj.c.';
                obj.C4 = obj.C2*obj.P*obj.c.';
                
            end
        end
        
        %slack update
        %the projection function II, returns new Z+, W+ matrixes
        function obj = II(obj)
            obj.Z = obj.X;
            obj.W = obj.U;
            
            %update X slack to stay below max magnitude
            for i = 1:size(obj.Z, 1)
                if norm(obj.Z(i, :), 2) <= -obj.xCon
                    obj.Z(i, :) = zeros(size(obj.Z), 2);
                elseif norm(obj.Z(i, :), 2) <= obj.xCon
                    obj.Z(i, :) = obj.Z(i, :);
                else
                    obj.Z(i, :) = .5*(1+ obj.xCon/norm(obj.Z(i, :), 2)) * obj.Z(i, :);
                end
            end

            %update U slack
            for i = 1:size(obj.W, 1)
                obj.W(i, 4) = norm(obj.W(i, 1:3), 2);
                
                %update U slack to stay below max magnitude
                if obj.W(i, 4) <= -obj.uMax
                    obj.W(i, 1:3) = zeros(3,1);
                elseif obj.W(i, 4) <= obj.uMax
                    
                    obj.W(i, :) = obj.W(i, :);
                else
                    obj.W(i, 1:3) = .5*(1+ obj.uMax/obj.W(i, 4)) * obj.W(i, 1:3);
                    
                end
                
                %update U slack to in cone angle
                if obj.W(i, 4)*cos(obj.phi) <= -obj.W(i, 3)
                    obj.W(i, 1:3) = zeros(3,1);

                elseif obj.W(i, 4)*cos(obj.phi) <= obj.W(i, 3)
                    
                    obj.W(i, :) = obj.W(i, :);
                else
                    obj.W(i, 1:3) = (obj.W(i, 3)/(obj.W(i, 4)*cos(obj.phi))) * obj.W(i, 1:3);
                    
                end
                
                %update U slack to have the slack in the slack to be
                %magnitude of thrust
                if obj.W(i, 4) <= norm(obj.W(i, 1:3), 2)
                    obj.W(i, 4) = obj.W(i, 4);
                else
                    obj.W(i, 4) = norm(obj.W(i, 1:3), 2);
                end

            end
        end

        
        
        %dual update (updates Lamda, Mu, and pentalty)
        function obj = dualUpdate(obj)
            for i = 1:size(obj.Lamda, 1)
                obj.Lamda(i, :) = obj.Lamda(i, :) + obj.penalty*(obj.X(i, :) - obj.Z(i, :));
            end

            for i = 1:size(obj.Mu, 1)
                obj.Mu(i, :) = obj.Mu(i, :) + obj.penalty*(obj.U(i, :) - obj.W(i, :));
            end
            obj.penalty = obj.penalty;
            
        end
        
        %runs LQR(false), then does the loop of: 1. LQR(true) {primal
        %update}; 2. II() {slack update}; 3. dualUpdate() {dual update}
        %for a certain number of iterations or until X/U stops changing by
        %a certain tolerence
        %update
        function obj = run(obj)
            obj = obj.LQR(false);
            obj = obj.LQR(true);
            obj = obj.II();
            obj = obj.dualUpdate();
            %delta = obj.U - obj.W;

            %e = 0;
            
            for e = 1:200 
                %e = e + 1;
                obj = obj.LQR(true);
                obj = obj.II();
                obj = obj.dualUpdate();
                %delta = obj.U - obj.W
               
            end
            
        end


    end
end
    