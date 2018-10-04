# -*- coding: utf-8 -*-

"""
This python module implements helpers for IPython / Jupyter and CUDA

Copyright (C) 2018  SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging

from IPython.core import magic_arguments
from IPython.core.magic import line_magic, Magics, magics_class
import pycuda.driver as cuda

import Timer, CudaContext


@magics_class
class MyIPythonMagic(Magics): 
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        'name', type=str, help='Name of context to create')
    @magic_arguments.argument(
        '--blocking', '-b', action="store_true", help='Enable blocking context')
    @magic_arguments.argument(
        '--no_cache', '-nc', action="store_true", help='Disable caching of kernels')
    def cuda_context_handler(self, line):
        args = magic_arguments.parse_argstring(self.cuda_context_handler, line)
        self.logger =  logging.getLogger(__name__)
        
        self.logger.info("Registering %s in user workspace", args.name)
        
        if args.name in self.shell.user_ns.keys():
            self.logger.debug("Context already registered! Ignoring")
            return
        else:
            self.logger.debug("Creating context")
            use_cache = False if args.no_cache else True
            self.shell.user_ns[args.name] = CudaContext.CudaContext(blocking=args.blocking, use_cache=use_cache)
        
        # this function will be called on exceptions in any cell
        def custom_exc(shell, etype, evalue, tb, tb_offset=None):
            self.logger.exception("Exception caught: Resetting to CUDA context %s", args.name)
            while (cuda.Context.get_current() != None):
                context = cuda.Context.get_current()
                self.logger.info("Popping <%s>", str(context.handle))
                cuda.Context.pop()

            if args.name in self.shell.user_ns.keys():
                self.logger.info("Pushing <%s>", str(self.shell.user_ns[args.name].cuda_context.handle))
                self.shell.user_ns[args.name].cuda_context.push()
            else:
                self.logger.error("No CUDA context called %s found (something is wrong)", args.name)
                self.logger.error("CUDA will not work now")

            self.logger.debug("==================================================================")
            
            # still show the error within the notebook, don't just swallow it
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offsetContext)

        # this registers a custom exception handler for the whole current notebook
        get_ipython().set_custom_exc((Exception,), custom_exc)
        
        
        # Handle CUDA context when exiting python
        import atexit
        def exitfunc():
            self.logger.info("Exitfunc: Resetting CUDA context stack")
            while (cuda.Context.get_current() != None):
                context = cuda.Context.get_current()
                self.logger.info("`-> Popping <%s>", str(context.handle))
                cuda.Context.pop()
            self.logger.debug("==================================================================")
        atexit.register(exitfunc)
        
        
        
        
        
        
        
        
    logger_initialized = False
    
    
    
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '--level', '-l', type=int, default=0, help='The level of logging to screen [0, 50]')
    def setup_logging(self, line):
        if (self.logger_initialized):
            logging.getLogger('').info("Global logger already initialized!")
            return;
        else:
            self.logger_initialized = True
            
            args = magic_arguments.parse_argstring(self.setup_logging, line)
            import sys
            
            #Get root logger
            logger = logging.getLogger('')
            logger.setLevel(args.level)

            #Add log to screen
            ch = logging.StreamHandler()
            ch.setLevel(args.level)
            logger.addHandler(ch)
            logger.log(args.level, "Console logger using level %s", logging.getLevelName(args.level))
        
        logger.info("Python version %s", sys.version)









# Register 
ip = get_ipython()
ip.register_magics(MyIPythonMagic)

