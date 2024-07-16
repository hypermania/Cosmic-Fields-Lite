#ifndef MIDPOINT_HPP
#define MIDPOINT_HPP



//#include <boost/numeric/odeint/stepper/base/explicit_stepper_base.hpp>
//#include <boost/numeric/odeint/algebra/range_algebra.hpp>
#include <boost/numeric/odeint/algebra/default_operations.hpp>
#include <boost/numeric/odeint/algebra/algebra_dispatcher.hpp>
#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resizer.hpp>

#ifndef DISABLE_CUDA
#include "cuda_wrapper.cuh"
#endif

template<
  class State,
  class Value = double,
  class Deriv = State,
  class Time = Value,
  class Algebra = typename boost::numeric::odeint::algebra_dispatcher<State>::algebra_type,
  class Operations = typename boost::numeric::odeint::operations_dispatcher<State>::operations_type,
  class Resizer = boost::numeric::odeint::initially_resizer //boost::numeric::odeint::always_resizer
  >
class midpoint : public boost::numeric::odeint::algebra_stepper_base<Algebra, Operations>
{  
public :
  typedef State state_type;
  typedef State deriv_type;
  typedef Value value_type;
  typedef Time time_type;
  typedef unsigned short order_type;
  typedef boost::numeric::odeint::stepper_tag stepper_category;

  typedef boost::numeric::odeint::algebra_stepper_base<Algebra, Operations> algebra_stepper_base_type;
  typedef typename algebra_stepper_base_type::algebra_type algebra_type;
  typedef typename algebra_stepper_base_type::operations_type operations_type;

  static order_type order(void) { return 2; }
  
  midpoint(){}

  template<class System>
  void do_step(System system, State &in, Time t, Time dt)
  {
    static const Value val1 = static_cast<Value>(1);
    const Time dh = dt / static_cast<Value>(2);
    const Time th = t + dh;

    //m_resizer.adjust_size(in, boost::numeric::odeint::detail::bind(&stepper_type::template resize_impl<State>, boost::numeric::odeint::detail::ref(*this), boost::numeric::odeint::detail::_1));
    m_resizer.adjust_size(in, [&](const auto &arg){ return resize_impl(arg); });
    
    typename boost::numeric::odeint::unwrap_reference<System>::type &sys = system;

    sys(in, deriv_tmp.m_v, t);
    algebra_stepper_base_type::m_algebra.for_each3(state_tmp.m_v, in, deriv_tmp.m_v,
						   typename operations_type::template scale_sum2<Value, Time>(val1, dh));
    
    sys(state_tmp.m_v, deriv_tmp.m_v, th);
    algebra_stepper_base_type::m_algebra.for_each3(state_tmp.m_v, in, deriv_tmp.m_v,
						   typename operations_type::template scale_sum2<Value, Time>(val1, dt));

    in.swap(state_tmp.m_v);

    // Release memory
    //m_resizer.adjust_size(State(), [&](const auto &arg){ return resize_impl(arg); });
    // deriv_tmp.m_v.clear();
    // State().swap(deriv_tmp.m_v);
    // state_tmp.m_v.clear();
    // State().swap(state_tmp.m_v);
  }

  // template<class StateType>
  // void adjust_size(const StateType &x)
  // {
  //   resize_impl(x);
  // }
  
  bool resize_impl(const State &x)
  {
    bool resized = false;
    resized |= boost::numeric::odeint::adjust_size_by_resizeability(deriv_tmp, x, typename boost::numeric::odeint::is_resizeable<State>::type());
    resized |= boost::numeric::odeint::adjust_size_by_resizeability(state_tmp, x, typename boost::numeric::odeint::is_resizeable<State>::type());
    return resized;
  }
  
private:
  Resizer m_resizer;

  boost::numeric::odeint::state_wrapper<State> deriv_tmp;
  boost::numeric::odeint::state_wrapper<State> state_tmp;
};



#endif
