program hello
  implicit none
  type(psb_ctxt_type) :: icontxt
  integer(psb_ipk_) ::  iam, np, ip, jp, idummy

  call psb_init(icontxt)
  call psb_info(icontxt,iam,np)            
  !   have all processes check in 
  if ((iam >= 0).and.(iam < np)) then 
    if (iam == 0)  then 
      do ip = 1, np-1
        call psb_rcv(icontxt,idummy,ip)          
      enddo
      write(*,*) 'Hello, world: all ',np, &
           & ' processes checked in!'
    else
      ip = 0
      call psb_snd(icontxt,idummy,ip)
    endif
  end if
  call psb_exit(icontxt)
end program hello