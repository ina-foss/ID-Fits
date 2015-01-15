#include <SnoopFaceDescLib/Stats.h>


ostream& operator<<(std::ostream& os, const PlotValues& plot_values)
{
    return plot_values.toStream(os);
}
