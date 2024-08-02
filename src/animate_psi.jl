# animation settings
for (L,g) in Iterators.product([8,10,12],[-0.5,-1.0,-2.0])
  # L = 10
  # g = -1.0
  J = -1.0

  time = Observable(1)
  dt = 0.01
  ts = collect(0:dt:2)

  psis = analytical_timeEv((L,J,g), ts)
  # psis = [abs.(psi) for psi in psis]
  psiRmax = findmax([findmax(abs.(real.(psi)))[1] for psi in psis])[1]
  psiImax = findmax([findmax(abs.(imag.(psi)))[1] for psi in psis])[1]
  psimax = max(psiRmax, psiImax)

  dataR = @lift(real.(psis[$time]))
  dataI = @lift(imag.(psis[$time]))

  fig, ax, p = scatter(dataR, dataI, color = RGBAf(0., 0.2, 0.8, 0.1),
                       axis = (title = @lift("t = $(round($time * dt, digits = 2))"), limits = ((-psimax,psimax),(-psimax,psimax)), ))

  timestamps = collect(1:length(ts))
  framerate = div(length(timestamps),4)

  record(fig, "/Users/wladi/Desktop/psiEvolGifs/psiEvol_L=$(L)_g=$(g).mp4", timestamps; framerate = framerate) do t
    time[] = t
  end
end
