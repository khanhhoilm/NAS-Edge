

def get_top_arch(args):
    # Lazy imports to avoid loading unnecessary dependencies
    if args.nasbench == "nasbench101":
        from .get_nasbench101 import get_top_arch as get_nasbench101_top_arch
        return get_nasbench101_top_arch(args)
    elif args.nasbench == "nasbench201":
        from .get_nasbench201 import get_top_arch as get_nasbench201_top_arch
        return get_nasbench201_top_arch(args)
    elif args.nasbench == "nasbench301":
        from .get_nasbench301 import get_top_arch as get_nasbench301_top_arch
        return get_nasbench301_top_arch(args)
    elif args.nasbench == "hwnasbench":
        from .get_hwnasbench import get_top_arch as get_hwnasbench_top_arch
        return get_hwnasbench_top_arch(args)
    elif args.nasbench == "accelnasbench":
        from .get_accelnasbench import get_top_arch as get_accelnasbench_top_arch
        return get_accelnasbench_top_arch(args)
    elif args.nasbench == "suitezero":
        from .get_suitezero import get_top_arch as get_suitezero_top_arch
        return get_suitezero_top_arch(args)
    elif args.nasbench == "nasbench_x11":
        from .get_nasbench_x11 import get_top_arch as get_nasbench_x11_top_arch
        return get_nasbench_x11_top_arch(args)
    elif args.nasbench == "nasbench_graph":
        from .get_nasbench_graph import get_top_arch as get_nasbench_graph_top_arch
        return get_nasbench_graph_top_arch(args)
    else:
        raise NotImplementedError(f"Top architecture retrieval not implemented for {args.nasbench}")