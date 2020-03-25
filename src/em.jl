"""
    update_em_stats!(estim_settings::EstimSettings, Xs::FloatVector, Xs_old::FloatVector, Ps::SymMatrix, Ps_old::SymMatrix, E::FloatArray, F::FloatArray, G::FloatArray, YXs::FloatMatrix)

Update the EM statistics.
"""
function update_em_stats!(estim_settings::EstimSettings, Xs::FloatVector, Xs_old::FloatVector, Ps::SymMatrix, Ps_old::SymMatrix, E::FloatArray, F::FloatArray, G::FloatArray, YXs::FloatMatrix)

    # Views
    Xs_view = @view Xs[1:estim_settings.s];
    Ps_view = @view Ps[1:estim_settings.s, 1:estim_settings.s];
    Xs_old_view = @view Xs_old[1:estim_settings.sp];
    Ps_old_view = @view Ps_old[1:estim_settings.sp, 1:estim_settings.sp];
    PPs_view = @view Ps[1:estim_settings.s, estim_settings.s+1:end];

    # Update EM statistics
    E .+= Xs_view*Xs_view' + Ps_view;
    F .+= Xs_view*Xs_old_view' + PPs_view;
    G .+= Xs_old_view*Xs_old_view' + Ps_old_view;
end

"""
    ksmoother_em!(estim_settings::EstimSettings, kalman_settings::MutableKalmanSettings, status::KalmanStatus)

Kalman smoother: RTS smoother from the last evaluated time period in status to t==0.

This implementation of the smoother returns the EM statistics and updates the initial conditions in KalmanSettings.

# Model
The state space model used below is,

``Y_{t} = B*X_{t} + e_{t}``

``X_{t} = C*X_{t-1} + v_{t}``

Where ``e_{t} ~ N(0, R)`` and ``v_{t} ~ N(0, V)``.

# Arguments
- `estim_settings`: EstimSettings struct used for the estimation.
- `kalman_settings`: MutableKalmanSettings struct.
- `status`: KalmanStatus struct.
"""
function ksmoother_em!(estim_settings::EstimSettings, kalman_settings::MutableKalmanSettings, status::KalmanStatus)

    # Memory pre-allocation
    E = zeros(estim_settings.s, estim_settings.s);
    F = zeros(estim_settings.s, estim_settings.sp);
    G = zeros(estim_settings.sp, estim_settings.sp);
    YXs = zeros(kalman_settings.n, kalman_settings.m);
    Xs = status.X_post;
    Ps = status.P_post;

    # Loop over t
    for t=status.t:-1:2

        # Pointers
        Xp = status.history_X_prior[t];
        Pp = status.history_P_prior[t];
        Xf_lagged = status.history_X_post[t-1];
        Pf_lagged = status.history_P_post[t-1];

        # Smoothed estimates for t-1
        J1 = compute_J1(Pf_lagged, Pp, kalman_settings);
        Xs_old = backwards_pass(Xf_lagged, J1, Xs, Xp);
        Ps_old = backwards_pass(Pf_lagged, J1, Ps, Pp);

        # Update EM statistics
        update_em_stats!(estim_settings, Xs, Xs_old, Ps, Ps_old, E, F, G);

        # Update Xs and Ps
        Xs = copy(Xs_old);
        Ps = copy(Ps_old);
    end

    # Pointers
    Xp = status.history_X_prior[1];
    Pp = status.history_P_prior[1];

    # Compute smoothed estimates for t==0
    J1 = compute_J1(kalman_settings.P0, Pp, kalman_settings);
    kalman_settings.X0 = backwards_pass(kalman_settings.X0, J1, Xs, Xp);
    kalman_settings.P0 = backwards_pass(kalman_settings.P0, J1, Ps, Pp);

    # Update EM statistics
    update_em_stats!(estim_settings, Xs, kalman_settings.X0, Ps, kalman_settings.P0, E, F, G);

    # Use Symmetric for G
    G_sym = Symmetric(G)::SymMatrix;

    # Return EM statistics
    return E, F, G_sym;
end

"""
"""
function em_observation!(kalman_settings::MutableKalmanSettings, estim_settings::EstimSettings, E::FloatMatrix)
end

"""
    em_transition!(kalman_settings::MutableKalmanSettings, estim_settings::EstimSettings, E::SymMatrix, F::FloatMatrix, G::SymMatrix)

Update coefficients of the transition equation.
"""
function em_transition!(kalman_settings::MutableKalmanSettings, estim_settings::EstimSettings, E::FloatMatrix, F::FloatMatrix, G::SymMatrix)

    # Initialise
    Ψ = @view kalman_settings.C[1:estim_settings.s, 1:estim_settings.sp];
    E_var = @view E[estim_settings.s, estim_settings.s];

    # VAR(p) coefficients
    for i=1:estim_settings.s
        F_i = @view F[i, 1:estim_settings.sp];
        kalman_settings.C[i, 1:estim_settings.sp] = inv(G)*F_i;
    end

    # Covariance matrix of the VAR(p) residuals
    parent(kalman_settings.V)[1:estim_settings.s, 1:estim_settings.s] = Symmetric(E_var-F*Ψ'-Ψ*F'+Ψ*G*Ψ')::SymMatrix ./ estim_settings.T;
end

"""
    em_routine(kalman_settings::MutableKalmanSettings, estim_settings::EstimSettings, loglik_old::Float64)

Run EM iteration.
"""
function em_routine(kalman_settings::MutableKalmanSettings, estim_settings::EstimSettings, loglik_old::Float64)

    # Run Kalman filter
    status = KalmanStatus();
    for t=1:kalman_settings.T
        kfilter!(kalman_settings, status);
    end

    # Converged
    converged = isconverged(status.loglik, loglik_old, estim_settings.tol, estim_settings.ε, true);

    # Check convergency
    if converged
        verb_message(estim_settings.verb, "em > converged!\n");

    # Not converged
    else

        # Sufficient statistics
        E, F, G = ksmoother_em!(kalman_settings, status);

        # Update coefficients: observation equation
        em_observation!();

        # Update coefficients: observation equation
        em_transition!(kalman_settings, estim_settings, E, F, G);
    end

    # Return output
    return converged, status.loglik;
end

"""
    em_estimation(kalman_settings::MutableKalmanSettings, estim_settings::EstimSettings)

Estimate state-space models via an EM algorithm.

The code implements the EM similarly to the ECM in Pellegrino (2020).

# Arguments
- `kalman_settings`: MutableKalmanSettings struct.
- `estim_settings`: EstimSettings struct used for the estimation.

# References
- Dempster, Laird and Rubin (1977), Pellegrino (2020), Shumway and Stoffer (1982).
"""
function em_estimation(kalman_settings::MutableKalmanSettings, estim_settings::EstimSettings)
end
